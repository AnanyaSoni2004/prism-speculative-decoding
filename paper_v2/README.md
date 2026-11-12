# Paper v2: Persistence, Not Variability

- `paper.pdf`, `paper.tex`: the rewritten paper. Tables are generated files (`tab_*.tex`); do not edit them by hand.
- `sim_models.py`: exact numbers for the synthetic Models A and C -> `sim_numbers.json`.
- `make_results.py`: reads the trace and result files in `../experiments/qwen`, computes the formula-error
  table with paired bootstrap intervals, parses the adaptive results, draws `fig_signflip.pdf` and `fig_error.pdf`
  -> `results.json`.
- `gen_tables.py`: writes `tab_formula.tex`, `tab_adaptive.tex`, `tab_modelc.tex` from `results.json`.

Rebuild: `python sim_models.py && python make_results.py && python gen_tables.py && pdflatex paper && pdflatex paper`.
To regenerate the result files it parses, run `adaptive_analysis.py` in `experiments/qwen` first (see the header of that script).

Numbers typed by hand in the text (check them if any data changes): the abstract, Section 3 "Two synthetic models",
Sections 4.1-4.2, the check against real decoding and Table 2 in Section 5.2, and Sections 6.1-6.3.
