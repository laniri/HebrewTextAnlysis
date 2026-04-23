import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MainLayout from '../components/MainLayout';

describe('Task 13.3: Responsive Layout', () => {
  it('should render MainLayout with flex-col and md:flex-row classes', () => {
    render(
      <MainLayout
        editorPane={<div data-testid="editor">Editor</div>}
        analysisPanel={<div data-testid="analysis">Analysis</div>}
      />,
    );

    // The root container should have both flex-col (mobile) and md:flex-row (desktop)
    const container = screen.getByTestId('editor').closest('.flex');
    expect(container).not.toBeNull();
    expect(container!.className).toContain('flex-col');
    expect(container!.className).toContain('md:flex-row');
  });

  it('should render editor pane with responsive width classes', () => {
    render(
      <MainLayout
        editorPane={<div data-testid="editor">Editor</div>}
        analysisPanel={<div data-testid="analysis">Analysis</div>}
      />,
    );

    // Editor section should have md:w-3/5 for desktop and min-h-[300px] for mobile
    const editorSection = screen.getByLabelText('עורך טקסט');
    expect(editorSection.className).toContain('md:w-3/5');
    expect(editorSection.className).toContain('min-h-[300px]');
    expect(editorSection.className).toContain('flex-1');
  });

  it('should render analysis panel with responsive width classes', () => {
    render(
      <MainLayout
        editorPane={<div data-testid="editor">Editor</div>}
        analysisPanel={<div data-testid="analysis">Analysis</div>}
      />,
    );

    // Analysis aside should have md:w-2/5 for desktop
    const analysisAside = screen.getByLabelText('לוח ניתוח');
    expect(analysisAside.className).toContain('md:w-2/5');
    expect(analysisAside.className).toContain('overflow-y-auto');
  });

  it('should render analysis panel with border-t for mobile and md:border-t-0 for desktop', () => {
    render(
      <MainLayout
        editorPane={<div data-testid="editor">Editor</div>}
        analysisPanel={<div data-testid="analysis">Analysis</div>}
      />,
    );

    const analysisAside = screen.getByLabelText('לוח ניתוח');
    // On mobile: border-t separates stacked panes
    // On desktop: md:border-t-0 removes the top border
    expect(analysisAside.className).toContain('border-t');
    expect(analysisAside.className).toContain('md:border-t-0');
  });

  it('should render both panes as children', () => {
    render(
      <MainLayout
        editorPane={<div data-testid="editor">Editor Content</div>}
        analysisPanel={<div data-testid="analysis">Analysis Content</div>}
      />,
    );

    expect(screen.getByTestId('editor')).toBeDefined();
    expect(screen.getByTestId('analysis')).toBeDefined();
    expect(screen.getByText('Editor Content')).toBeDefined();
    expect(screen.getByText('Analysis Content')).toBeDefined();
  });

  it('should have proper aria labels for accessibility', () => {
    render(
      <MainLayout
        editorPane={<div>Editor</div>}
        analysisPanel={<div>Analysis</div>}
      />,
    );

    // Hebrew aria labels for the two panes
    expect(screen.getByLabelText('עורך טקסט')).toBeDefined();
    expect(screen.getByLabelText('לוח ניתוח')).toBeDefined();
  });
});
