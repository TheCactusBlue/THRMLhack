import type { Action } from "../types";

interface ActionQueueProps {
  actions: Action[];
  onUndo: () => void;
  onClearAll: () => void;
  onCommit: () => void;
  disabled?: boolean;
}

export function ActionQueue({
  actions,
  onUndo,
  onClearAll,
  onCommit,
  disabled = false,
}: ActionQueueProps) {
  if (actions.length === 0) {
    return null;
  }

  return (
    <div className="bg-neutral-800 border-2 border-neutral-700 rounded-lg p-3 mb-3 max-w-md mx-auto">
      <h3 className="text-sm font-bold text-gray-300 mb-2 flex items-center gap-2">
        📋 Planned Actions
        <span className="text-xs text-gray-500">({actions.length})</span>
      </h3>

      <div className="max-h-32 overflow-y-auto mb-2 space-y-1">
        {actions.map((action, idx) => (
          <div
            key={action.id}
            className="text-xs bg-neutral-900 px-2 py-1.5 rounded border border-neutral-700 flex items-center gap-2"
          >
            <span className="text-gray-500 font-mono">{idx + 1}.</span>
            <span className="flex-1 text-gray-300">{action.description}</span>
            <span className="text-gray-500">
              {action.type === "bias" ? "⚡" : "🔗"}
            </span>
          </div>
        ))}
      </div>

      <div className="flex gap-2 justify-between">
        <button
          onClick={onUndo}
          disabled={disabled || actions.length === 0}
          className="px-3 py-1.5 text-xs font-semibold bg-neutral-700 hover:bg-neutral-600 rounded text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          ↩ Undo Last
        </button>

        <button
          onClick={onClearAll}
          disabled={disabled || actions.length === 0}
          className="px-3 py-1.5 text-xs font-semibold bg-red-600/80 hover:bg-red-600 rounded text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          🗑 Clear All
        </button>

        <button
          onClick={onCommit}
          disabled={disabled || actions.length === 0}
          className="px-4 py-1.5 text-xs font-bold bg-emerald-600 hover:bg-emerald-500 rounded text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
        >
          ✓ COMMIT
        </button>
      </div>
    </div>
  );
}
