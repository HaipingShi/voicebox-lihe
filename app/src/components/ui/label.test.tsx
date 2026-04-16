import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Label } from './label';

describe('Label', () => {
  it('renders text content', () => {
    render(<Label>Test Label</Label>);
    expect(screen.getByText('Test Label')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<Label className="custom-class">Label</Label>);
    const label = screen.getByText('Label');
    expect(label).toHaveClass('custom-class');
  });

  it('applies default label styles', () => {
    render(<Label>Styled</Label>);
    const label = screen.getByText('Styled');
    expect(label).toHaveClass('text-sm');
    expect(label).toHaveClass('font-medium');
  });

  it('passes through HTML attributes', () => {
    render(<Label htmlFor="input-id">With For</Label>);
    const label = screen.getByText('With For');
    expect(label).toHaveAttribute('for', 'input-id');
  });
});
