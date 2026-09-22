"""Print a rendered prompt for one example. `python scripts/show_prompt.py <task> [prompt] [split]`"""
import sys
from ptx.tasks import get_task

task = get_task(sys.argv[1])
prompts = task.base_prompts()
name = sys.argv[2] if len(sys.argv) > 2 else next(iter(prompts))
split = sys.argv[3] if len(sys.argv) > 3 else "test"
spec = prompts[name]
ex = task.load(split)[0]
r = task.render(ex, spec)
print(f"task={task.name} prompt={name} id={spec.prompt_id} fmt_sensitive={task.format_sensitive}")
print(f"source={spec.provenance.get('source')}")
print(f"stop={r.stop}")
print("-" * 72)
print(r.text)
print("-" * 72)
print(f"GOLD: {ex.gold!r}")
