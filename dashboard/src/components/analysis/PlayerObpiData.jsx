"use client";

import { useMemo, useState } from "react";

const PAGE_SIZE = 12;
const minuteOptions = [10, 20, 30, 40, 50, 60, 70, 80, 90];

export function PlayerObpiData({ metrics = [] }) {
  if (!metrics.length) {
    return (
      <section className="card p-5">
        <h2 className="text-lg font-semibold text-white">StatsBomb OBPI data</h2>
        <p className="mt-2 text-sm text-muted">No grouped OBPI event data is available for this player.</p>
      </section>
    );
  }

  return (
    <section className="card p-5">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-white">StatsBomb OBPI data</h2>
        <p className="mt-1 text-sm text-muted">Extracted match event columns grouped by OBPI metric.</p>
      </div>

      <div className="space-y-4">
        {metrics.map((metric) => (
          <MetricGroup key={metric.key} metric={metric} />
        ))}
      </div>
    </section>
  );
}

function MetricGroup({ metric }) {
  const [valueFilter, setValueFilter] = useState("all");
  const [minuteFilter, setMinuteFilter] = useState("all");
  const [page, setPage] = useState(1);

  const filteredRows = useMemo(() => {
    return (metric.rows || []).filter((row) => {
      if (valueFilter === "non-null" && !metric.columns.every((column) => hasDisplayValue(row[column]))) {
        return false;
      }

      if (minuteFilter !== "all" && !isInsideMinuteWindow(row.minute, Number(minuteFilter))) {
        return false;
      }

      return true;
    });
  }, [metric, minuteFilter, valueFilter]);

  const pageCount = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  const pageRows = filteredRows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  function updateValueFilter(value) {
    setValueFilter(value);
    setPage(1);
  }

  function updateMinuteFilter(value) {
    setMinuteFilter(value);
    setPage(1);
  }

  return (
    <div className="rounded-md border border-slate-700 bg-slate-900 p-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-white">
            {metric.key} {metric.label}
          </h3>
          <p className="mt-1 text-xs text-muted">
            {metric.row_count || 0} matching events
          </p>
        </div>
      </div>

      <div className="mt-3 text-xs text-slate-300">
        <span className="text-muted">Columns: </span>
        {metric.columns?.join(", ")}
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <label className="text-xs font-medium text-slate-300">
          Row values
          <select
            value={valueFilter}
            onChange={(event) => updateValueFilter(event.target.value)}
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          >
            <option value="all">Show All</option>
            <option value="non-null">Show Non-null</option>
          </select>
        </label>
        <label className="text-xs font-medium text-slate-300">
          Match time
          <select
            value={minuteFilter}
            onChange={(event) => updateMinuteFilter(event.target.value)}
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          >
            <option value="all">Show All</option>
            {minuteOptions.map((minute) => (
              <option key={minute} value={minute}>
                {minute === 10 ? "0-10" : `${minute - 9}-${minute}`}
              </option>
            ))}
          </select>
        </label>
      </div>

      <MetricRows metric={metric} rows={pageRows} filteredCount={filteredRows.length} />
      <MetricPagination currentPage={currentPage} pageCount={pageCount} onPageChange={setPage} />
    </div>
  );
}

function MetricRows({ metric, rows, filteredCount }) {
  if (!rows.length) {
    return <p className="mt-3 text-xs text-muted">No rows match the current filters.</p>;
  }

  return (
    <div className="mt-3 overflow-x-auto">
      <div className="mb-2 text-xs text-muted">Showing {rows.length} of {filteredCount} filtered rows</div>
      <table className="min-w-full border-collapse text-left text-xs">
        <thead>
          <tr className="border-b border-slate-700 text-muted">
            {metric.columns.map((column) => (
              <th key={column} className="whitespace-nowrap px-2 py-2 font-medium">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${metric.key}-${index}`} className="border-b border-slate-800 last:border-0">
              {metric.columns.map((column) => (
                <td key={column} className="max-w-48 truncate px-2 py-2 text-slate-200" title={formatValue(row[column])}>
                  {formatValue(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricPagination({ currentPage, pageCount, onPageChange }) {
  if (pageCount <= 1) return null;

  return (
    <div className="mt-3 flex items-center justify-between gap-3 text-xs text-slate-300">
      <button
        type="button"
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
        className="rounded-md border border-slate-700 px-3 py-2 disabled:cursor-not-allowed disabled:opacity-50"
      >
        Previous
      </button>
      <span>
        Page {currentPage} of {pageCount}
      </span>
      <button
        type="button"
        onClick={() => onPageChange(Math.min(pageCount, currentPage + 1))}
        disabled={currentPage === pageCount}
        className="rounded-md border border-slate-700 px-3 py-2 disabled:cursor-not-allowed disabled:opacity-50"
      >
        Next
      </button>
    </div>
  );
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "null";
  if (Array.isArray(value) || typeof value === "object") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function hasDisplayValue(value) {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim() !== "";
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

function isInsideMinuteWindow(rowMinute, selectedMinute) {
  const minute = Number(rowMinute);
  if (!Number.isFinite(minute)) return false;
  const min = selectedMinute === 10 ? 0 : selectedMinute - 9;
  return minute >= min && minute <= selectedMinute;
}
