import { describe, expect, it } from 'vitest';
import { formatDuration, formatEngineName, formatFileSize } from './format';

describe('formatDuration', () => {
  it('formats seconds into mm:ss', () => {
    expect(formatDuration(65)).toBe('1:05');
  });

  it('pads single-digit seconds', () => {
    expect(formatDuration(9)).toBe('0:09');
  });

  it('handles zero', () => {
    expect(formatDuration(0)).toBe('0:00');
  });

  it('handles large values', () => {
    expect(formatDuration(3661)).toBe('61:01');
  });
});

describe('formatEngineName', () => {
  it('returns display name for known engines', () => {
    expect(formatEngineName('qwen')).toBe('Qwen');
    expect(formatEngineName('luxtts')).toBe('LuxTTS');
    expect(formatEngineName('chatterbox')).toBe('Chatterbox');
    expect(formatEngineName('chatterbox_turbo')).toBe('Chatterbox Turbo');
  });

  it('appends model size for qwen', () => {
    expect(formatEngineName('qwen', 'Small')).toBe('Qwen Small');
  });

  it('defaults to Qwen for undefined engine', () => {
    expect(formatEngineName(undefined)).toBe('Qwen');
  });

  it('passes through unknown engine names', () => {
    expect(formatEngineName('custom')).toBe('custom');
  });
});

describe('formatFileSize', () => {
  it('formats bytes', () => {
    expect(formatFileSize(500)).toBe('500 Bytes');
  });

  it('formats KB', () => {
    expect(formatFileSize(1024)).toBe('1 KB');
  });

  it('formats MB', () => {
    expect(formatFileSize(1048576)).toBe('1 MB');
  });

  it('formats GB', () => {
    expect(formatFileSize(1073741824)).toBe('1 GB');
  });

  it('handles zero', () => {
    expect(formatFileSize(0)).toBe('0 Bytes');
  });
});
