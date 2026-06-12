import type { Page } from '@playwright/test'

export const MOCK_USER = {
  user: { login: 'testuser', name: 'Test User', avatar_url: '' },
  auth_enabled: false,
}

export const MOCK_SESSION = {
  id: 'sess-e2e-001',
  title: 'E2E test session',
  user_login: 'testuser',
  created_at: '2026-06-12T00:00:00',
  updated_at: '2026-06-12T00:00:00',
  messages: [],
}

export const MOCK_PROJECTS = [
  {
    id: 'jertel_elastalert2',
    repo_name: 'jertel/elastalert2',
    repo_url: 'https://github.com/jertel/elastalert2',
    status: 'indexed',
    progress_pct: 100,
    node_count: 630,
    edge_count: 3930,
    file_count: 64,
    last_indexed_at: '2026-06-12T18:00:00',
    created_at: '2026-06-12T17:00:00',
    updated_at: '2026-06-12T18:00:00',
  },
  {
    id: 'nicholasgibson2_elastalert-jertel',
    repo_name: 'nicholasgibson2/elastalert-jertel',
    repo_url: 'https://github.com/nicholasgibson2/elastalert-jertel',
    status: 'indexed',
    progress_pct: 100,
    node_count: 301,
    edge_count: 503,
    file_count: 16,
    last_indexed_at: '2026-06-12T17:00:00',
    created_at: '2026-06-12T16:00:00',
    updated_at: '2026-06-12T17:00:00',
  },
]

export const MOCK_REVIEW = {
  id: 'rev-e2e-001',
  project_id: 'jertel_elastalert2',
  pr_url: 'https://github.com/jertel/elastalert2/pull/1706',
  repo_name: 'jertel/elastalert2',
  pr_number: 1706,
  status: 'completed',
  created_at: '2026-06-12T18:00:00',
  updated_at: '2026-06-12T18:30:00',
  total_patches: 1,
  author_login: 'nsano-rururu',
  file_reviews: [
    {
      file_path: 'elastalert/prometheus_wrapper.py',
      phase1_found: true,
      phase2_fault: 'metrics_writeback (line 37) — does not handle potential KeyError',
      patch: '--- a/elastalert/prometheus_wrapper.py\n+++ b/elastalert/prometheus_wrapper.py\n@@ -35,3 +35,5 @@\n def metrics_writeback(self, data):\n-    value = data["value"]\n+    value = data.get("value")\n+    if value is None:\n+        return\n',
      applies_cleanly: true,
      eval_score: 4,
      risk_level: 'medium',
    },
  ],
}

export async function setupMockApi(page: Page) {
  await page.route('/auth/me', route =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) })
  )
  await page.route('/api/projects', route =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_PROJECTS) })
  )
  await page.route('/api/reviews', route =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([MOCK_REVIEW]) })
  )
  await page.route(`/api/reviews/${MOCK_REVIEW.id}`, route =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_REVIEW) })
  )
  await page.route('/api/chat/sessions', async route => {
    if (route.request().method() === 'GET') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) })
    } else {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_SESSION) })
    }
  })
  await page.route('/api/chat/sessions/**', route =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_SESSION) })
  )
}

export function makeSseResponse(events: object[]): string {
  return events.map(e => `data: ${JSON.stringify(e)}\n\n`).join('')
}
