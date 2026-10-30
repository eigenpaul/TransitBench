# TransitBench base image: the mini preset's agent and grading boxes (docs/IMAGE.md).
#   docker build -t transitbench-base:local -f image/Dockerfile image
#   prime images push transitbench-base:m3 --context image --dockerfile image/Dockerfile --private
# It holds no TransitBench code and no data: task setup writes the code (ENV_DESIGN D13), and
# Dockerfile.dockerignore keeps context/ (the dev set) out of the build context.
FROM python:3.12-slim-bookworm@sha256:2090ee9f64d462a36c0c76309a869bae63f3c722470f96030bd14a825c88d0b2

COPY install.sh requirements.txt /tmp/tb-image/
RUN sh /tmp/tb-image/install.sh transitbench-base:m3 && rm -rf /tmp/tb-image

WORKDIR /app
