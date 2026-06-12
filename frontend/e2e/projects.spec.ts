import { test, expect } from '@playwright/test'
import { setupMockApi, MOCK_PROJECTS, MOCK_REVIEW } from './helpers'

test.describe('Projects page', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockApi(page)
    await page.goto('/projects')
  })

  test('shows project list', async ({ page }) => {
    await expect(page.getByText('jertel/elastalert2')).toBeVisible()
    await expect(page.getByText('nicholasgibson2/elastalert-jertel')).toBeVisible()
  })

  test('shows indexed status for all projects', async ({ page }) => {
    const badges = page.getByText('Indexed')
    await expect(badges.first()).toBeVisible()
  })

  test('shows node counts', async ({ page }) => {
    await expect(page.getByText(/630/)).toBeVisible()
    await expect(page.getByText(/301/)).toBeVisible()
  })
})

test.describe('Review detail page', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockApi(page)
    await page.goto(`/reviews/${MOCK_REVIEW.id}`)
  })

  test('shows PR number and repo name', async ({ page }) => {
    await expect(page.getByText('jertel/elastalert2')).toBeVisible()
    await expect(page.getByText(/#1706/)).toBeVisible()
  })

  test('shows completed status', async ({ page }) => {
    await expect(page.getByText(/completed/i)).toBeVisible()
  })

  test('shows patch count', async ({ page }) => {
    await expect(page.getByText(/1/)).toBeVisible()
  })

  test('shows fault description', async ({ page }) => {
    await expect(page.getByText(/prometheus_wrapper\.py/)).toBeVisible()
  })
})
