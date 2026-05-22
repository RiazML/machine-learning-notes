name: 🐛 Bug Report
about: Create a report to help us improve the study guide or fix code errors.
title: "[BUG] "
labels: bug
assignees: ""

body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to report a bug or typo!
  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear and concise description of what the bug/typo is.
      placeholder: E.g., The DBSCAN flowchart contains a typo, or Day 13 code throws a NameError.
    validations:
      required: true
  - type: input
    id: file-affected
    attributes:
      label: Affected File(s)
      description: Which file(s) contain the issue?
      placeholder: E.g., 132_dbscan_clustering_algorithms.md
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Error Details / Steps to Reproduce
      description: Paste any error traceback, invalid syntax snippets, or steps to reproduce here.
      placeholder: |
        ```python
        # Paste code or error message here
        ```
    validations:
      required: false
  - type: textarea
    id: proposed-fix
    attributes:
      label: Proposed Fix
      description: If you know how to fix it, describe it here!
      placeholder: E.g., Change A[("placement.csv")] to A["placement.csv"].
    validations:
      required: false
