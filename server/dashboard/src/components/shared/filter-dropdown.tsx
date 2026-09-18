"use client";

import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

export interface FilterOption {
  value: string;
  /** How many rows carry this value under the other active filters. */
  count: number;
}

interface FilterDropdownProps {
  options: FilterOption[];
  /** Selected value; "" means no filter. */
  value: string;
  onChange: (value: string) => void;
  /** Trigger and first-item label while nothing is selected, e.g. "All projects". */
  allLabel: string;
  searchPlaceholder: string;
  className?: string;
}

// Lists this short are quicker to scan than to search.
const SEARCH_FROM = 8;

export function FilterDropdown({
  options,
  value,
  onChange,
  allLabel,
  searchPlaceholder,
  className,
}: FilterDropdownProps) {
  const [open, setOpen] = useState(false);

  const choose = (next: string) => {
    onChange(next === value ? "" : next);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn("justify-between px-3 font-normal", className)}
        >
          <span className="truncate">{value || allLabel}</span>
          <ChevronsUpDown className="ml-2 size-3.5 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] min-w-64 p-0"
      >
        <Command>
          {options.length >= SEARCH_FROM && (
            <CommandInput placeholder={searchPlaceholder} />
          )}
          <CommandList>
            <CommandEmpty>No matches.</CommandEmpty>
            <CommandGroup>
              <CommandItem value="__all__" onSelect={() => choose("")}>
                <Check
                  className={cn(
                    "mr-2 size-4 shrink-0",
                    value === "" ? "opacity-100" : "opacity-0",
                  )}
                />
                {allLabel}
              </CommandItem>
              {options.map((opt) => (
                <CommandItem
                  key={opt.value}
                  value={opt.value}
                  onSelect={() => choose(opt.value)}
                >
                  <Check
                    className={cn(
                      "mr-2 size-4 shrink-0",
                      opt.value === value ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <span className="truncate">{opt.value}</span>
                  <span className="ml-auto pl-2 text-xs text-onSurface-default-tertiary">
                    {opt.count}
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
