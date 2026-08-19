# Privacy note for the user-study release

`user_study_data.csv` contains participant-level repeated-measures ratings. It
is pseudonymized, not anonymous and not aggregated.

The public table contains no names, Prolific IDs, IP addresses, timestamps,
free-text responses, or lookup table linking participant codes to real-world
identities. Exact age, gender, and AI-usage-frequency fields were removed from
the public release after a disclosure-risk review because their combinations
could single out participants within this small sample. Those fields are not
needed to reproduce the paper's specified mixed-effects model.

Participant codes exist only to associate the three repeated ratings from the
same participant. Do not attempt re-identification or linkage. Use of the file
is governed by `HUMAN-DATA-USE-NOTICE.txt`, not CC BY 4.0.

Raw Prolific exports and the code-to-identity/payment mapping are not included
in this repository. Requests for additional research data require separate
ethical and institutional review.

The historical consent text uses the word “anonymous.” This release uses the
more precise term “pseudonymized” because stable participant codes remain for
the repeated-measures model, even though the repository contains no code-to-
identity mapping. The consent file is retained without silently rewriting what
participants saw; its separate statement allowing de-identified open-science
sharing is the applicable description of this release.
