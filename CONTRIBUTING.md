# Contributing to Machine Learning Notes

Thank you for your interest in improving these Machine Learning Notes! Contributions from the community help make this resource more accurate, comprehensive, and accessible for everyone.

---

## Code of Conduct

By contributing, you agree to keep interactions respectful, professional, and welcoming for all participants.

---

## How Can I Contribute?

### 1. Reporting Typos or Bugs
If you spot an equation typo, markdown formatting issue, or python code error:
1. Open a **Bug Report** issue.
2. Provide details about the affected file and the error.

### 2. Proposing Content Improvements
If you want to add new explanations, extra visualizations, or new topics:
1. Open a **Feature Request** issue to discuss the proposal.
2. Once aligned, submit a Pull Request following the guidelines below.

---

## Pull Request Guidelines

To maintain code and markdown consistency, please follow these guidelines:

### Formatting Rules
1. **Title Prefix**: Do not add `"Day X:"` or `"Day X -"` prefixes to any H1 headings. All files should start with a clean H1: `# [Topic Title]`.
2. **Left-Aligned Python Blocks**: Make sure all Python code blocks inside markdown files are fully left-aligned (no leading indentation block-wide).
3. **Quoted Mermaid Labels**: If a Mermaid flowchart node contains special characters (like parentheses, colons, arithmetic operators, brackets, etc.), you must wrap the label in double quotes, e.g.:
   - **Correct**: `A["Regression (Continuous)"]`
   - **Incorrect**: `A[Regression (Continuous)]`
4. **Offline Runnable Code**: Ensure that code blocks are fully executable offline and match Scikit-Learn parity wherever comparisons are made.

### Running Local Validation
Before submitting your Pull Request, run the local audit test suite to verify everything passes:

```bash
python tests/test_guide.py
```

If the audit reports any failures (such as unquoted Mermaid labels, invalid Python syntax, or empty files), fix them before pushing your changes.
