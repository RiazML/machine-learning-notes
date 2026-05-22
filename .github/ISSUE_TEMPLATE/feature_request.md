name: 💡 Feature Request
about: Propose a new topic, math explanation, visualization, or code tutorial.
title: "[FEATURE] "
labels: enhancement
assignees: ""

body:
  - type: markdown
    attributes:
      value: |
        Have an idea to make these machine learning notes even better? Tell us about it!
  - type: textarea
    id: feature-description
    attributes:
      label: Describe your idea
      description: A clear and concise description of what you want to add or improve.
      placeholder: E.g., Add notes on Support Vector Regression (SVR) or extra visualizations for PCA.
    validations:
      required: true
  - type: textarea
    id: rationale
    attributes:
      label: Rationale
      description: Why would this addition be helpful for other learners?
      placeholder: E.g., SVR is a widely used algorithm and completing the SVM modules would cover all bases.
    validations:
      required: true
  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context, references, or links here.
    validations:
      required: false
