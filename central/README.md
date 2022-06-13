# DJL - Central Server

## Overview

This module contains the DJLCentral web interface implementation.

## Documentation

You can build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Run model server

Use the following command to start model server locally:

```sh
cd serving/central

# for Linux/macOS:
./gradlew run

# for Windows:
..\..\gradlew run
```

The DJLCentral server will be listening on port 8080.

open your browser and type in url 

```ssh
http://localhost:8080/
```
