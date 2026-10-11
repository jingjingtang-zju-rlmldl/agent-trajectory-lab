# Contributions

Run `python -m unittest discover -s tests -v`. New diagnostics should include a minimal counterexample and a non-failing control. Never add a claimed model result without a reproducible command, data provenance, model revision, verifier configuration and raw outcomes.

Keep private trajectories, credentials and model weights out of commits. The built-in fixtures are deliberately synthetic. A real trainer adapter should be pinned to a tested version and must not infer missing masks from text length.
