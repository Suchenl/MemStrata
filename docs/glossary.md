# Terminology (unified terms, aligned with the abstract/intro)

Principle: **any concept without a novel contribution uses a generic term; only a
genuine contribution gets a proper name.** The entire text (method / related /
experiments / conclusion / appendix / figures & tables / notation) must be
consistent with the abstract and introduction.

| Level | Canonical term | Notes / counter-examples |
| --- | --- | --- |
| Method name | **MemStrata** (macro `\method`, = **Mem**ory **Strata**) | Do not use "StrataMem" / "AutoFilm" / "Montage" anymore (Montage is the project/repo name, not the method name) |
| Core mechanism | **active composition over role-aware assets** | Turns passive retrieval into active composition; do not call it "memory retrieval" |
| Core object | **role-aware asset** (role = conditioning function: identity anchor / scene / style / motion / negative …) | This is a contribution, keep it |
| Persistent state | **asset bank** $\mathcal{A}_n$ | Generic term; **do not** use "production memory" / "production asset space" |
| A single asset | **asset** $a_j$, containing multiple **purpose-specific representations** $\mathcal{R}_j$ | — |
| Asset relations | **typed relations** ($\rho_j$, folded into the asset) | **Do not** list "asset graph $\mathcal{G}_n$" as a separate component |
| Global constraints | **global constraints / text assets** | **Do not** use "production bible $\mathcal{B}$"; these are just a set of text/spec assets |
| Raw retention | **archive** (full film / raw output / logs / slices) | Generic term; do not call it "production archive" |
| Per-step generation conditioning | **Composed Context** $\mathcal{C}_n$ | Contribution; **do not** use "asset package" / "composition package" |
| Composition method | **model-free composition** (deterministic dereference, no model call) | — |
| Lifecycle | **lifecycle status** $\psi$: candidate/reusable/used/rejected/deprecated/failed | — |
| Update mechanism | **condition-aware curation** | A generic description is enough, no proper name needed |
| Generator | **reference-conditioned continuation generator** (black-box, generator-agnostic) | — |
| Evaluation | **MemStrata-Bench**, four axes: asset selection / functional-role assignment / constraint satisfaction / negative-asset avoidance | Do not use "asset composition evaluation" as a proper name |

Deprecated legacy terms (must no longer appear anywhere in the text): StrataMem,
production memory, production bible, asset relation graph $\mathcal{G}_n$, asset
package / composition package, production archive, AutoFilm, LVG-Agent, and the
LVG-Agent-era latent/memory-record notation ($m_n=(k_n,v_n,u_n,a_n)$,
$\hat{\mathcal{R}}_n$, etc.).
