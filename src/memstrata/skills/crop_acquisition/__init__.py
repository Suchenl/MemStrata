"""Self-contained crop-acquisition subsystem (S5 propose/identify/novelty perception).

Vendored (copy + import-rewrite) from the benchmark's
``vmem_bench.annotation.pipeline.stages.s5_entities_visual_crop_acquisition`` so the
production ``memstrata`` package obeys its hard rule: **zero imports of ``vmem_bench``**
(``memstrata/docs/design_philosophy.md`` §5, ``benchmarks/MemStrata/AGENTS.md`` rule 2).

Public entry points (import submodules directly to keep heavy deps lazy):
  * ``orchestrator.acquire_entity_crop`` — targeted per-entity, novelty-first acquisition
    (the requested path, O_req).
  * ``discovery.discover_entities`` — type-constrained proposals for entities nobody
    named (the discovery path, O_disc), including the location concept vocabulary the
    requested path lacks.
  * ``crop_client.ProposeIdentifyCropper`` — production ``Cropper`` protocol adapter.
  * ``crop_client.ServerConceptDiscoverer`` — production ``Discoverer`` adapter, reusing
    the cropper's server so discovery costs no extra model load.
  * ``crop_server`` — persistent GDINO+SAM3+DINOv3 file-queue server (``job_kind`` =
    ``acquire`` | ``discover``).

Heavy model imports (torch / transformers / SAM3) are confined to the model-load path
(``crop_server`` / the ``_ensure_loaded`` methods), so importing the client + orchestrator
does not require transformers>=5.9.
"""
