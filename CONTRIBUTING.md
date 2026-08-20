#### Documentation

The code is the documentation. `docs/08_api_reference.md` is **generated** from
the docstrings in `pycute/`, so document a name where it is defined and let the
test suite rewrite the reference -- it regenerates a stale one and warns that it
did, leaving you to commit the result. To regenerate it by hand, or to check it
the way CI does:

```bash
$ python scripts/gen_api_reference.py           # rewrite the reference
$ python scripts/gen_api_reference.py --check   # what CI runs
```

A docstring is a one-line summary, then any prose that says something the
conditions and examples cannot, then the sections that carry the contract:

```python
def coalesce(A, profile=1, *, mode=()):
  """
  Coalesce a Layout or Tensor into a simpler, equivalent form.

  Pre-conditions:
    conditions the caller must meet
  Post-conditions:
    size(result) == size(A)
  Examples:
    coalesce(Layout((2, (1, 6)), (1, (6, 2))))  == Layout(12, 1)
    coalesce(bad)                               -> ValueError
  """
```

Prefer conditions and examples to prose: they say the same thing more precisely,
and unlike prose they are checked. `test/test_docstring_examples.py` evaluates
every line under an `Examples:` heading -- `==` as an assertion, `->` as an
expected exception, anything else as a setup statement sharing a namespace with
the lines that follow. Write a schematic illustration with `:=` and keep it in
prose, where it will not be evaluated.

#### Signing Off Your Work
* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
  * Any contribution which contains commits that are not Signed-Off will not be accepted.
* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```
* Full text of the DCO (https://developercertificate.org/):
  ```
    Developer Certificate of Origin
    Version 1.1
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.
    Developer's Certificate of Origin 1.1
    By making a contribution to this project, I certify that:
    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or
    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or
    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.
    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
