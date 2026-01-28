import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'COMPLETED':
      return 'bg-green-500';
    case 'RUNNING':
      return 'bg-blue-500';
    case 'FAILED':
      return 'bg-red-500';
    case 'PENDING':
      return 'bg-yellow-500';
    case 'CANCELLED':
      return 'bg-gray-500';
    default:
      return 'bg-gray-400';
  }
}

export function getCategoryColor(category: string): string {
  const colors: Record<string, string> = {
    commonsense: '#8884d8',
    deontology: '#82ca9d',
    justice: '#ffc658',
    virtue: '#ff7300',
    mixed: '#00C49F',
  };
  return colors[category.toLowerCase()] || '#8884d8';
}
