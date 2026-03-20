## Task State Updates

- Use `state_update` only when task state actually changes.
- A valid update must change at least one of:
  `overall_goal`, `current_task`, `task_plan`, `current_step_id`,
  `completed_steps`, `blocked_reason`, or `todos`.
- Valid moments include:
  creating/revising a plan, moving to a new step, marking a step completed,
  or blocker/todo changes.
- Do not call `state_update` repeatedly with unchanged values.
- Do not use `state_update` as a pre-action ritual before every tool call.
