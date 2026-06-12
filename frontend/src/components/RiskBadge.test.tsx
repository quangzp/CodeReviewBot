import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import RiskBadge from './RiskBadge'

describe('RiskBadge', () => {
  it('renders "Low Risk" for level=low', () => {
    render(<RiskBadge level="low" />)
    expect(screen.getByText(/Low Risk/i)).toBeTruthy()
  })

  it('renders "Medium Risk" for level=medium', () => {
    render(<RiskBadge level="medium" />)
    expect(screen.getByText(/Medium Risk/i)).toBeTruthy()
  })

  it('renders "High Risk" for level=high', () => {
    render(<RiskBadge level="high" />)
    expect(screen.getByText(/High Risk/i)).toBeTruthy()
  })

  it('renders checkmark icon for low risk', () => {
    render(<RiskBadge level="low" />)
    expect(screen.getByText('✓')).toBeTruthy()
  })

  it('renders warning icon for medium risk', () => {
    render(<RiskBadge level="medium" />)
    expect(screen.getByText('⚠')).toBeTruthy()
  })

  it('renders x icon for high risk', () => {
    render(<RiskBadge level="high" />)
    expect(screen.getByText('✗')).toBeTruthy()
  })

  it('applies green style for low risk', () => {
    const { container } = render(<RiskBadge level="low" />)
    expect(container.firstChild).toHaveClass('bg-green-100')
  })

  it('applies yellow style for medium risk', () => {
    const { container } = render(<RiskBadge level="medium" />)
    expect(container.firstChild).toHaveClass('bg-yellow-100')
  })

  it('applies red style for high risk', () => {
    const { container } = render(<RiskBadge level="high" />)
    expect(container.firstChild).toHaveClass('bg-red-100')
  })

  it('normalizes uppercase input', () => {
    render(<RiskBadge level="HIGH" />)
    expect(screen.getByText(/High Risk/i)).toBeTruthy()
  })

  it('shows unknown style for unrecognized level', () => {
    const { container } = render(<RiskBadge level="extreme" />)
    expect(container.firstChild).toHaveClass('bg-gray-100')
  })

  it('shows question mark icon for unknown level', () => {
    render(<RiskBadge level="extreme" />)
    expect(screen.getByText('?')).toBeTruthy()
  })
})
