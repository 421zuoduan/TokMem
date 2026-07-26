ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG APT_SANDBOX_USER=_apt

RUN apt-get -o APT::Sandbox::User="${APT_SANDBOX_USER}" update && \
    if [ "${APT_SANDBOX_USER}" = "root" ]; then \
        mkdir -p /tmp/singleuid-bin && \
        printf '#!/bin/sh\nexit 0\n' > /tmp/singleuid-bin/chown && \
        cp /tmp/singleuid-bin/chown /tmp/singleuid-bin/chgrp && \
        cp /tmp/singleuid-bin/chown /tmp/singleuid-bin/dpkg-statoverride && \
        chmod 755 /tmp/singleuid-bin/chown /tmp/singleuid-bin/chgrp /tmp/singleuid-bin/dpkg-statoverride && \
        cp /usr/bin/chown /tmp/singleuid-bin/chown.real && \
        cp /usr/bin/chgrp /tmp/singleuid-bin/chgrp.real && \
        cp /usr/bin/dpkg-statoverride /tmp/singleuid-bin/dpkg-statoverride.real && \
        cp /tmp/singleuid-bin/chown /usr/bin/chown && \
        cp /tmp/singleuid-bin/chgrp /usr/bin/chgrp && \
        cp /tmp/singleuid-bin/dpkg-statoverride /usr/bin/dpkg-statoverride && \
        DEBIAN_FRONTEND=noninteractive apt-get -o APT::Sandbox::User=root install -y bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree net-tools iputils-ping coreutils curl cpio jq && \
        cp /tmp/singleuid-bin/chown.real /usr/bin/chown && \
        cp /tmp/singleuid-bin/chgrp.real /usr/bin/chgrp && \
        cp /tmp/singleuid-bin/dpkg-statoverride.real /usr/bin/dpkg-statoverride && \
        rm -rf /tmp/singleuid-bin; \
    else \
        DEBIAN_FRONTEND=noninteractive apt-get install -y bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree net-tools iputils-ping coreutils curl cpio jq; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG FILE_SYSTEM_VERSION
ENV file_system_version=${FILE_SYSTEM_VERSION} \
    script=setup_nl2b_fs_${FILE_SYSTEM_VERSION}.sh

COPY docker /tmp/intercode-docker
RUN test -n "${FILE_SYSTEM_VERSION}" && \
    cp "/tmp/intercode-docker/bash_scripts/${script}" "/${script}" && \
    chmod +x "/${script}" && \
    "/${script}" && \
    cp /tmp/intercode-docker/docker.gitignore /.gitignore && \
    rm -rf /tmp/intercode-docker && \
    git config --global user.email "intercode@pnlp.org" && \
    git config --global user.name "intercode" && \
    git init && \
    git add -A && \
    git commit -m "initial commit"

WORKDIR /
