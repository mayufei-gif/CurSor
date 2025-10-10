# GitHub Actions Log Access Attempt

- Requested log URL: https://github.com/mayufei-gif/CurSor/actions/runs/18409562127/job/52457933604?pr=5
- Retrieval method: `curl -L` from the container environment.
- Result: Received the generic Actions job HTML page, but the actual job logs are gated behind authentication. The downloaded HTML displays "Sign in to view logs" instead of the step output, so the log contents cannot be inspected without valid GitHub credentials.

Because this environment does not have authenticated GitHub access, further analysis of the workflow output is blocked. If log review is required, please supply access credentials or provide an exported log artifact that is publicly accessible.
