FROM carlasim/carla:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /carla-simulator-cam

COPY carla-simulator-cam carla-simulator-cam/
RUN apt-get -y update
RUN apt-get install -y python3-pip
RUN apt-get install -y x11-apps
RUN pip3 install -r requirements.txt

CMD ["bash"]