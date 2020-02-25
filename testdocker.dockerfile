# This file tells docker what image must be created
# in order to be ahble to test this library
FROM  ubuntu:18.04

# base packages
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y --no-install-recommends -o Dpkg::Options::="--force-confnew"
# Install numpy and scipy
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends -o Dpkg::Options::="--force-confnew" \
                    python3-pip git
RUN pip3 install setuptools
RUN pip3 install numpy
RUN pip3 install sympy
RUN pip3 install scipy
# user handling
ARG myuser
ARG myuid
ARG mygroup
ARG mygid
ARG scriptdir
RUN addgroup --gid ${mygid} ${mygroup}
RUN adduser --gecos "" --disabled-password  --uid ${myuid} --gid ${mygid} ${myuser}
#add user to sudoers
RUN echo "${myuser} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
WORKDIR  /test
RUN chown ${myuser}:${mygroup} /test
