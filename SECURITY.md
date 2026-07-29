# Security policy

Do not report credentials in a public issue. Contact the project maintainers
privately using the security contact that will be added with the paper release.

Never commit tokens, `.env` files, participant information, restricted NSD or
COCO data, server addresses, or private filesystem paths. API credentials must
be supplied through environment variables. Generated outputs and checkpoints
are excluded by `.gitignore`.
