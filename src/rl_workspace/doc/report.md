---
documentclass: article
title: Reinforcement Learning for Vision-based Manipulation
subtitle: Autobots VIP Fall 2023
author: Anthony Miyaguchi, Ritesh Mehta, and Abdullatif Rashdan
date: 2023-12-08
colorlinks: true
dpi: 300
graphics: yes
geometry:
  - margin=0.8in
bibliography: report.bibtex
# https://stackoverflow.com/a/58840456
header-includes: |
  \usepackage{subfig}
  \usepackage{float}
  \let\origfigure\figure
  \let\endorigfigure\endfigure
  \renewenvironment{figure}[1][2] {
      \expandafter\origfigure\expandafter[H]
  } {
      \endorigfigure
  }
---

# Overview

We use the `gymnasium-robotics` environment to train a 6-DOF robotic arm. [@gymnasium_robotics2023github]

# Background

## Reinforcement Learning

## Vision-based Manipulation

# Approach

# Experiments

# Conclusions

# References
