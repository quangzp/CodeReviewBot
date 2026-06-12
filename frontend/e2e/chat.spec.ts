import { test, expect } from '@playwright/test'
import { setupMockApi, MOCK_SESSION, MOCK_PROJECTS, makeSseResponse } from './helpers'

test.describe('Chat interface', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockApi(page)
    await page.goto('/')
  })

  test('shows chat input', async ({ page }) => {
    await expect(page.getByPlaceholder(/ask anything/i)).toBeVisible()
  })

  test('shows send button', async ({ page }) => {
    await expect(page.getByRole('button', { name: /send/i })).toBeVisible()
  })

  test('sends message and receives text response', async ({ page }) => {
    await page.route('/api/chat', route =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: makeSseResponse([
          { type: 'session', id: MOCK_SESSION.id, title: 'Test' },
          { type: 'thinking', step: 0 },
          { type: 'message', text: 'Found 2 projects: jertel/elastalert2 (indexed), nicholasgibson2/elastalert-jertel (indexed).' },
          { type: 'done' },
        ]),
      })
    )

    await page.getByPlaceholder(/ask anything/i).fill('What projects do I have?')
    await page.getByRole('button', { name: /send/i }).click()

    await expect(page.getByText(/Found 2 projects/)).toBeVisible({ timeout: 10_000 })
  })

  test('renders project list card from tool result', async ({ page }) => {
    await page.route('/api/chat', route =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: makeSseResponse([
          { type: 'session', id: MOCK_SESSION.id, title: 'Test' },
          { type: 'tool_call', name: 'list_projects', args: {} },
          {
            type: 'tool_result',
            name: 'list_projects',
            summary: 'Found 2 project(s)',
            render: { kind: 'project_list', projects: MOCK_PROJECTS },
          },
          { type: 'message', text: 'You have 2 indexed projects.' },
          { type: 'done' },
        ]),
      })
    )

    await page.getByPlaceholder(/ask anything/i).fill('List my projects')
    await page.getByRole('button', { name: /send/i }).click()

    await expect(page.getByText('jertel/elastalert2')).toBeVisible({ timeout: 10_000 })
  })

  test('shows error message when review fails', async ({ page }) => {
    await page.route('/api/chat', route =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: makeSseResponse([
          { type: 'session', id: MOCK_SESSION.id, title: 'Test' },
          { type: 'error', message: 'Something went wrong' },
        ]),
      })
    )

    await page.getByPlaceholder(/ask anything/i).fill('Review a PR')
    await page.getByRole('button', { name: /send/i }).click()

    await expect(page.getByText(/something went wrong/i)).toBeVisible({ timeout: 10_000 })
  })

  test('send button disabled when input is empty', async ({ page }) => {
    const send = page.getByRole('button', { name: /send/i })
    await expect(send).toBeDisabled()

    await page.getByPlaceholder(/ask anything/i).fill('hello')
    await expect(send).toBeEnabled()
  })
})
