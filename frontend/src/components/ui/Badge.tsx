import { badgeStyle } from "@/lib/utils";

export function Badge({ name }: { name: string }) {
  const style = badgeStyle(name);
  const label = name.replace("-", " ").replace("mastery", "\u2605");
  return (
    <span className={`badge ${style.bg} ${style.text}`}>
      {label}
    </span>
  );
}
