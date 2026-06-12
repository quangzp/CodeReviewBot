import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import StatusBadge from './StatusBadge'

describe('StatusBadge', () => {
  it('capitalizes and displays the status text', () => {
    render(<StatusBadge status="completed" />)
    expect(screen.getByText('Completed')).toBeTruthy()
  })

  it('applies green style for completed', () => {
    const { container } = render(<StatusBadge status="completed" />)
    expect(container.firstChild).toHaveClass('bg-green-100')
  })

  it('applies red style for failed', () => {
    const { container } = render(<StatusBadge status="failed" />)
    expect(container.firstChild).toHaveClass('bg-red-100')
  })

  it('applies blue style for running', () => {
    const { container } = render(<StatusBadge status="running" />)
    expect(container.firstChild).toHaveClass('bg-blue-100')
  })

  it('applies gray style for pending', () => {
    const { container } = render(<StatusBadge status="pending" />)
    expect(container.firstChild).toHaveClass('bg-gray-100')
  })

  it('applies green style for indexed', () => {
    const { container } = render(<StatusBadge status="indexed" />)
    expect(container.firstChild).toHaveClass('bg-green-100')
  })

  it('falls back to gray style for unknown status', () => {
    const { container } = render(<StatusBadge status="mystery-status" />)
    expect(container.firstChild).toHaveClass('bg-gray-100')
  })

  it('renders a status dot element', () => {
    const { container } = render(<StatusBadge status="running" />)
    const spans = container.querySelectorAll('span')
    expect(spans.length).toBeGreaterThan(1)
  })

  it('dot pulses for running status', () => {
    const { container } = render(<StatusBadge status="running" />)
    const dot = container.querySelector('span span')
    expect(dot?.className).toContain('animate-pulse')
  })
})
