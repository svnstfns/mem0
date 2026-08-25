"use client";

import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import { format } from "date-fns";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import DeleteConfirmationModal from "@/components/ui/delete-confirmation-modal";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { UpgradeBanner } from "@/components/self-hosted/upgrade-banner";
import { toast } from "@/components/ui/use-toast";
import { getErrorMessage } from "@/lib/error-message";
import { api } from "@/utils/api";
import { MEMORY_ENDPOINTS } from "@/utils/api-endpoints";
import { useApiQuery } from "@/hooks/use-api-query";
import { Memory } from "@/types/api";

const PAGE_SIZE = 20;
// Keep in sync with ALL_MEMORIES_LIMIT in server/main.py.
const MEMORY_FETCH_LIMIT = 5000;

function metaStr(m: Memory, key: string): string {
  const v = m.metadata?.[key];
  return typeof v === "string" ? v : "";
}

interface RelatedItem {
  id: string;
  snippet: string | null;
  kind: string | null;
  project: string | null;
  expired: boolean | null;
  relation: string;
  outgoing: boolean;
  confidence: number | null;
}

function metaTopics(m: Memory): string[] {
  const v = m.metadata?.topics;
  return Array.isArray(v) ? v.map(String) : [];
}

export default function MemoriesPage() {
  const [userId, setUserId] = useState("");
  const [kindFilter, setKindFilter] = useState("all");
  const [projectFilter, setProjectFilter] = useState("");
  const [topicFilter, setTopicFilter] = useState("");
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [memoryToDelete, setMemoryToDelete] = useState<Memory | null>(null);
  const [page, setPage] = useState(0);
  const [related, setRelated] = useState<RelatedItem[]>([]);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";

  const {
    data: memories = [],
    isLoading,
    refetch,
  } = useApiQuery<Memory[]>(
    async () => {
      const params = userId.trim()
        ? { user_id: userId.trim(), top_k: MEMORY_FETCH_LIMIT }
        : { top_k: MEMORY_FETCH_LIMIT };
      const res = await api.get(MEMORY_ENDPOINTS.BASE, { params });
      const raw = res.data?.results ?? res.data ?? [];
      return Array.isArray(raw) ? raw : [];
    },
    { errorToast: "Failed to load memories", initialData: [] },
  );

  const kinds = Array.from(
    new Set(memories.map((m) => metaStr(m, "kind")).filter(Boolean)),
  ).sort();
  const filteredMemories = memories.filter((m) => {
    if (kindFilter !== "all" && metaStr(m, "kind") !== kindFilter) return false;
    const project = projectFilter.trim().toLowerCase();
    if (project) {
      const own = metaStr(m, "project") || metaStr(m, "session_project");
      if (!own.toLowerCase().includes(project)) return false;
    }
    const topic = topicFilter.trim().toLowerCase();
    if (topic && !metaTopics(m).some((t) => t.toLowerCase().includes(topic)))
      return false;
    return true;
  });
  useEffect(() => {
    if (!selectedMemory) {
      setRelated([]);
      return;
    }
    let cancelled = false;
    api
      .get(`/memories/${selectedMemory.id}/related`, { params: { depth: 1 } })
      .then((res) => {
        if (!cancelled) setRelated(res.data?.results ?? []);
      })
      .catch(() => {
        if (!cancelled) setRelated([]);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedMemory]);

  const totalPages = Math.ceil(filteredMemories.length / PAGE_SIZE);
  const paginatedMemories = filteredMemories.slice(
    page * PAGE_SIZE,
    (page + 1) * PAGE_SIZE,
  );

  const handleDelete = async () => {
    if (!memoryToDelete) return;
    try {
      await api.delete(MEMORY_ENDPOINTS.BY_ID(memoryToDelete.id));
      toast({ title: "Memory deleted", variant: "success" });
      if (selectedMemory?.id === memoryToDelete.id) setSelectedMemory(null);
      setMemoryToDelete(null);
      void refetch();
    } catch (error) {
      toast({
        title: "Failed to delete memory",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const columns = [
    {
      key: "memory" as keyof Memory,
      label: "Content",
      width: 400,
      render: (value: string) => (
        <span className="line-clamp-2 text-sm">{value}</span>
      ),
    },
    {
      key: "metadata" as keyof Memory,
      label: "Kind",
      width: 90,
      render: (_value: Memory[keyof Memory], row: Memory) =>
        metaStr(row, "kind") ? (
          <Badge variant="secondary" className="text-xs">
            {metaStr(row, "kind")}
          </Badge>
        ) : (
          <span className="text-xs text-onSurface-default-tertiary">--</span>
        ),
    },
    {
      key: "metadata" as keyof Memory,
      label: "Topics",
      width: 140,
      render: (_value: Memory[keyof Memory], row: Memory) => (
        <span className="flex flex-wrap gap-1">
          {metaTopics(row)
            .slice(0, 3)
            .map((t) => (
              <Badge key={t} variant="outline" className="text-[10px]">
                {t}
              </Badge>
            ))}
        </span>
      ),
    },
    {
      key: "metadata" as keyof Memory,
      label: "Project",
      width: 120,
      render: (_value: Memory[keyof Memory], row: Memory) => (
        <span className="text-xs">
          {metaStr(row, "project") ||
            (metaStr(row, "scope") === "global" ? "global" : "--")}
        </span>
      ),
    },
    {
      key: "created_at" as keyof Memory,
      label: "Created",
      width: 120,
      render: (value: string) =>
        value ? format(new Date(value), "MMM d, yyyy") : "--",
    },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold font-fustat">Memories</h1>

      {memories.length >= MEMORY_FETCH_LIMIT && (
        <UpgradeBanner
          id="memories-1k"
          message="1,000+ memories stored. Categories can help organize them."
          ctaLabel="Explore Cloud"
          ctaUrl="https://app.mem0.ai?utm_source=oss&utm_medium=dashboard-memories"
          variant="cloud"
        />
      )}

      <div className="flex gap-3">
        <Input
          placeholder="Filter by User ID (optional)"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              setPage(0);
              refetch();
            }
          }}
          className="w-64"
        />
        <Select
          value={kindFilter}
          onValueChange={(v) => {
            setKindFilter(v);
            setPage(0);
          }}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Kind" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All kinds</SelectItem>
            {kinds.map((k) => (
              <SelectItem key={k} value={k}>
                {k}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Input
          placeholder="Filter by project"
          value={projectFilter}
          onChange={(e) => {
            setProjectFilter(e.target.value);
            setPage(0);
          }}
          className="w-48"
        />
        <Input
          placeholder="Filter by topic"
          value={topicFilter}
          onChange={(e) => {
            setTopicFilter(e.target.value);
            setPage(0);
          }}
          className="w-40"
        />
      </div>

      {isLoading ? (
        <TableSkeleton rows={5} columns={4} />
      ) : memories.length === 0 ? (
        <EmptyState
          title="No memories yet"
          description="Create your first memory by sending a POST /memories request."
        >
          <pre className="text-xs text-left bg-surface-default-secondary p-3 rounded font-mono overflow-x-auto mt-3 max-w-lg">
            {`curl -X POST ${apiUrl}/memories \\
  -H "X-API-Key: <your-key>" \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "I like hiking"}], "user_id": "alice"}'`}
          </pre>
          <a
            href="https://docs.mem0.ai/open-source/features/rest-api#memory-operations"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-onSurface-default-tertiary underline underline-offset-4 hover:text-onSurface-default-primary mt-2"
          >
            REST API reference
          </a>
        </EmptyState>
      ) : (
        <>
          <Card className="border-memBorder-primary overflow-hidden">
            <DataTable
              data={paginatedMemories}
              columns={columns}
              getRowKey={(row) => row.id}
              onRowClick={(row) => setSelectedMemory(row)}
              getRowClassName={(row) =>
                selectedMemory?.id === row.id
                  ? "bg-surface-default-tertiary"
                  : undefined
              }
            />
          </Card>
          {totalPages > 1 && (
            <div className="flex items-center justify-between text-sm text-onSurface-default-tertiary">
              <span>
                {page * PAGE_SIZE + 1}–
                {Math.min((page + 1) * PAGE_SIZE, filteredMemories.length)} of{" "}
                {filteredMemories.length}
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      )}

      <Sheet
        open={!!selectedMemory}
        onOpenChange={(open) => {
          if (!open) setSelectedMemory(null);
        }}
      >
        <SheetContent className="sm:max-w-md">
          <SheetHeader>
            <SheetTitle>Memory Detail</SheetTitle>
            <SheetDescription className="sr-only">
              View memory content and metadata
            </SheetDescription>
          </SheetHeader>
          {selectedMemory && (
            <div className="mt-6 space-y-4">
              <div className="space-y-1">
                <Label className="text-xs text-onSurface-default-tertiary">
                  Content
                </Label>
                <p className="text-sm">{selectedMemory.memory}</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <Label className="text-xs text-onSurface-default-tertiary">
                    ID
                  </Label>
                  <p className="text-xs font-mono break-all">
                    {selectedMemory.id}
                  </p>
                </div>
                {selectedMemory.user_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      User
                    </Label>
                    <p className="text-sm">{selectedMemory.user_id}</p>
                  </div>
                )}
                {selectedMemory.agent_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Agent
                    </Label>
                    <p className="text-sm">{selectedMemory.agent_id}</p>
                  </div>
                )}
                {selectedMemory.created_at && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Created
                    </Label>
                    <p className="text-sm">
                      {new Date(selectedMemory.created_at).toLocaleString()}
                    </p>
                  </div>
                )}
              </div>
              {selectedMemory.metadata &&
                Object.keys(selectedMemory.metadata).length > 0 && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Metadata
                    </Label>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {Object.entries(selectedMemory.metadata).map(([k, v]) => (
                        <p key={k} className="text-xs break-all">
                          <span className="text-onSurface-default-tertiary">
                            {k}:{" "}
                          </span>
                          {String(v)}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              {related.length > 0 && (
                <div className="space-y-1">
                  <Label className="text-xs text-onSurface-default-tertiary">
                    Related ({related.length})
                  </Label>
                  <div className="space-y-1">
                    {related.map((r) => {
                      const target = memories.find((m) => m.id === r.id);
                      return (
                        <button
                          key={`${r.relation}-${r.id}`}
                          type="button"
                          disabled={!target}
                          onClick={() => target && setSelectedMemory(target)}
                          className="block w-full text-left text-xs hover:bg-surface-default-tertiary rounded p-1"
                        >
                          <Badge variant="outline" className="text-[10px] mr-1">
                            {r.outgoing ? `${r.relation} →` : `← ${r.relation}`}
                          </Badge>
                          <span className={r.expired ? "line-through opacity-60" : ""}>
                            {r.snippet ?? r.id}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              <Button
                variant="outline"
                size="sm"
                className="text-onSurface-danger-primary"
                onClick={() => setMemoryToDelete(selectedMemory)}
              >
                <Trash2 className="size-3.5 mr-1" />
                Delete memory
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>

      <DeleteConfirmationModal
        isOpen={!!memoryToDelete}
        onClose={() => setMemoryToDelete(null)}
        onConfirm={handleDelete}
        title="Delete memory"
        description="This memory will be permanently removed. This cannot be undone."
        itemName={memoryToDelete?.id ?? ""}
        confirmButtonText="Delete"
      />
    </div>
  );
}
