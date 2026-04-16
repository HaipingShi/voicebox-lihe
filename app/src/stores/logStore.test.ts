import { describe, expect, it } from 'vitest';
import { useLogStore } from './logStore';
import type { ServerLogEntry } from '@/platform/types';

function mockEntry(overrides?: Partial<ServerLogEntry>): ServerLogEntry {
  return {
    stream: 'stdout',
    line: 'test message',
    ...overrides,
  };
}

describe('useLogStore', () => {
  it('starts with empty entries', () => {
    const store = useLogStore.getState();
    expect(store.entries).toEqual([]);
  });

  it('adds an entry', () => {
    useLogStore.setState({ entries: [] });
    useLogStore.getState().addEntry(mockEntry({ line: 'hello' }));

    const entries = useLogStore.getState().entries;
    expect(entries).toHaveLength(1);
    expect(entries[0].line).toBe('hello');
  });

  it('assigns incremental id and timestamp', () => {
    useLogStore.setState({ entries: [] });
    useLogStore.getState().addEntry(mockEntry());
    useLogStore.getState().addEntry(mockEntry());

    const entries = useLogStore.getState().entries;
    expect(entries[1].id).toBeGreaterThan(entries[0].id);
    expect(entries[1].timestamp).toBeGreaterThanOrEqual(entries[0].timestamp);
  });

  it('clears all entries', () => {
    useLogStore.setState({ entries: [] });
    useLogStore.getState().addEntry(mockEntry());
    useLogStore.getState().addEntry(mockEntry());

    useLogStore.getState().clear();
    expect(useLogStore.getState().entries).toEqual([]);
  });

  it('trims entries when exceeding MAX_LOG_ENTRIES (2000)', () => {
    useLogStore.setState({ entries: [] });
    for (let i = 0; i < 2001; i++) {
      useLogStore.getState().addEntry(mockEntry({ line: `msg ${i}` }));
    }

    const result = useLogStore.getState().entries;
    expect(result.length).toBeLessThanOrEqual(2000);
  });
});
