"""
Two-Model Pipeline — Planner/Worker Architecture

A practical inference pattern discovered during testing:

  Qwen 3.8 (177B, slow, smart)  →  Planner + Reviewer
  Qwen 3.6 (35B, fast, obedient)  →  Worker

The large model (3.8) reads the problem, writes a plan broken into
small exact steps, and reviews the results. The small model (3.6)
executes the plan at 3× speed.

This pipeline leverages the strengths of each model:
  - 3.8: Catches traps, reasons about specs, writes reports
  - 3.6: Fast execution, follows exact instructions, doesn't second-guess

The trade-off: swapping models costs ~25 seconds per handoff. This
is acceptable for long jobs but annoying for chat. For chat, just
keep the 3.6 loaded.

This module implements the pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PipelineRole(Enum):
    """Role of a model in the pipeline."""

    PLANNER = "planner"      # Reads problem, writes plan
    WORKER = "worker"        # Executes the plan
    REVIEWER = "reviewer"    # Reviews results, writes report


@dataclass
class PlanStep:
    """A single step in a model-generated plan."""

    step_id: int
    description: str
    input_refs: list[str] = field(default_factory=list)
    output_ref: str = ""
    expected_check: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from the two-model pipeline."""

    plan: list[PlanStep]
    worker_output: dict[str, str]
    review: str
    total_time_seconds: float
    plan_time_seconds: float
    worker_time_seconds: float
    review_time_seconds: float
    swap_overhead_seconds: float = 0.0


class TwoModelPipeline:
    """
    Orchestrates the Planner/Worker/Reviewer pipeline.

    The pipeline works as follows:
      1. PLANNER (3.8): Read problem → Write plan (small, exact steps)
      2. SWAP: Swap 3.8 out, load 3.6 (~25 seconds)
      3. WORKER (3.6): Execute each plan step at 3× speed
      4. SWAP: Swap 3.6 out, load 3.8 (~25 seconds)
      5. REVIEWER (3.8): Review results, write report

    For chat, skip the pipeline and keep 3.6 loaded.
    """

    def __init__(
        self,
        model_38_config: Optional[dict] = None,
        model_36_config: Optional[dict] = None,
        swap_cost_seconds: float = 25.0,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            model_38_config: Configuration for the 177B model.
            model_36_config: Configuration for the 35B model.
            swap_cost_seconds: Time to swap models in/out of memory.
        """
        self.model_38_config = model_38_config or {}
        self.model_36_config = model_36_config or {}
        self.swap_cost_seconds = swap_cost_seconds

        # Pipeline state
        self._current_model: Optional[str] = None
        self._current_plan: list[PlanStep] = []
        self._results: dict[str, str] = {}

    def run_pipeline(
        self,
        problem: str,
    ) -> PipelineResult:
        """
        Run the full Planner/Worker/Reviewer pipeline.

        Args:
            problem: The problem description to solve.

        Returns:
            PipelineResult with plan, output, and review.
        """
        import time

        start_time = time.time()

        # Step 1: Planner (3.8) — read problem, write plan
        plan_time_start = time.time()
        plan = self._plan(problem)
        plan_time = time.time() - plan_time_start

        # Step 2: Swap — unload 3.8, load 3.6
        swap_1 = self.swap_model("3.6")

        # Step 3: Worker (3.6) — execute plan
        worker_time_start = time.time()
        worker_output = self._execute_plan(plan)
        worker_time = time.time() - worker_time_start

        # Step 4: Swap — unload 3.6, load 3.8
        swap_2 = self.swap_model("3.8")

        # Step 5: Reviewer (3.8) — review results, write report
        review_time_start = time.time()
        review = self._review(plan, worker_output)
        review_time = time.time() - review_time_start

        total_time = time.time() - start_time

        return PipelineResult(
            plan=plan,
            worker_output=worker_output,
            review=review,
            total_time_seconds=total_time,
            plan_time_seconds=plan_time,
            worker_time_seconds=worker_time,
            review_time_seconds=review_time,
            swap_overhead_seconds=swap_1 + swap_2,
        )

    def chat_mode(self, prompt: str) -> str:
        """
        Chat mode — keep 3.6 loaded, no pipeline overhead.

        For interactive chat, skip the pipeline and just use the
        fast 35B model. It doesn't catch traps, but it's fast.

        Args:
            prompt: User prompt.

        Returns:
            Model response.
        """
        if self._current_model != "3.6":
            self.swap_model("3.6")
        return self._chat_with_model_36(prompt)

    def swap_model(self, model_name: str) -> float:
        """
        Swap a model in or out of memory.

        Args:
            model_name: "3.8" to load the 177B model, "3.6" for 35B.

        Returns:
            Time taken for the swap.
        """
        import time

        start = time.time()
        self._current_model = model_name
        elapsed = time.time() - start

        # In reality, this involves:
        # 1. Unloading the current model from VRAM/RAM
        # 2. Loading the new model's weights
        # 3. Initializing the model's state
        # On the tested hardware, this takes ~25 seconds

        return elapsed

    def _plan(self, problem: str) -> list[PlanStep]:
        """
        Planner (3.8): Read problem, write plan.

        The 177B model reads the problem specification, identifies
        traps and edge cases, and writes a plan broken into small,
        exact steps.

        Args:
            problem: Problem description.

        Returns:
            List of plan steps.
        """
        # Simulate the planner's analysis
        steps = [
            PlanStep(
                step_id=1,
                description="Read and parse the specification document",
                output_ref="parsed_spec",
            ),
            PlanStep(
                step_id=2,
                description="Identify all test cases and their expected behavior",
                output_ref="test_cases",
            ),
            PlanStep(
                step_id=3,
                description="Check for contradictions between spec and tests",
                output_ref="contradictions",
                expected_check="spec_priority",
            ),
            PlanStep(
                step_id=4,
                description="Identify and quarantine legacy/decoy files",
                output_ref="quarantined_files",
            ),
            PlanStep(
                step_id=5,
                description="Implement fixes according to spec (not tests)",
                output_ref="fixed_code",
            ),
            PlanStep(
                step_id=6,
                description="Write randomized property-based tests from spec",
                output_ref="property_tests",
            ),
            PlanStep(
                step_id=7,
                description="Run all tests and verify all pass",
                output_ref="test_results",
            ),
        ]
        self._current_plan = steps
        return steps

    def _execute_plan(
        self, plan: list[PlanStep]
    ) -> dict[str, str]:
        """
        Worker (3.6): Execute each plan step.

        The 35B model executes the plan at 3× speed. It follows exact
        instructions and doesn't second-guess.

        Args:
            plan: Plan steps from the planner.

        Returns:
            Dictionary of step outputs.
        """
        output = {}
        for step in plan:
            # Simulate execution (in reality, this runs actual code)
            output[step.output_ref] = self._execute_step(step)
        self._results = output
        return output

    def _execute_step(self, step: PlanStep) -> str:
        """Execute a single plan step."""
        # Simulate the worker's execution
        if step.step_id == 3:
            return "CONTRADICTION FOUND: Test #3 says request 3 of 3 should be denied, but spec says request at limit should be allowed. Spec takes priority."
        elif step.step_id == 4:
            return "QUARANTINED: legacy_limiter.py (2 bugs found), notes.md (flaky test reference), config.py (float trap)"
        elif step.step_id == 5:
            return "FIXED: Only the single config value that needs to be a fraction was changed. Others remain as whole numbers."
        elif step.step_id == 6:
            return "PROPERTY TESTS: 18000 randomized decisions generated. 0 mismatches with spec."
        elif step.step_id == 7:
            return "ALL TESTS PASS: 7/7 tests pass. Hidden checks verified."
        return "Step completed"

    def _review(
        self, plan: list[PlanStep], worker_output: dict[str, str]
    ) -> str:
        """
        Reviewer (3.8): Review results, write report.

        The 177B model reads the worker's output and writes a report,
        catching any issues the worker might have missed.

        Args:
            plan: Original plan.
            worker_output: Worker's output.

        Returns:
            Review report.
        """
        return (
            "Review Report:\n"
            "  Plan was followed correctly.\n"
            "  Contradictions between spec and tests were resolved in favor of spec.\n"
            "  Legacy/decoy files were quarantined, not modified.\n"
            "  Property-based tests confirm correctness.\n"
            "  All tests pass (7/7, including hidden checks).\n"
            "  Score: 7/7 — Full credit."
        )

    def _chat_with_model_36(self, prompt: str) -> str:
        """Simulate chat with the 35B model."""
        return f"[3.6 Chat] Response to: {prompt[:50]}..."

    def summary(self) -> str:
        """Human-readable pipeline summary."""
        return (
            f"Two-Model Pipeline:\n"
            f"  Planner: Qwen 3.8 (177B) — reads, plans, reviews\n"
            f"  Worker:  Qwen 3.6 (35B) — executes at 3× speed\n"
            f"  Swap cost: {self.swap_cost_seconds:.0f}s per handoff\n"
            f"  Use for: Long jobs (plan → execute → review)\n"
            f"  Skip for: Chat (keep 3.6 loaded)"
        )
