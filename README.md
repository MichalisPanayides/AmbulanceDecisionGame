[![Tests][tests-shield]][tests-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![DOI][doi-shield]][doi-url]

Ambulance Decision Game: A python library that attempts to explore a game theoretic approach to the EMS - ED interface
=======================================================================


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#tests">Tests</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project revolves around modelling the interaction between the Emergency Medical Service (EMS) and Emergency Departments. 

The `ambulance_game` library is used to model:
* a discrete event simulation model of a queueing system with two waiting spaces, where individuals can be blocked.
* the equivalent Markov chain model,
* a game theoretic model between 3 players; 2 queueing systems (EDs) and a distrbutor (EMS).


## Installation

Install a development version of this library with the command:

    $ python -m pip install flit
    $ python -m flit install --symlink


<!-- TESTS EXAMPLES -->
## Tests

Run all tests developed by first installing and then running `tox`:

    $ python -m pip install tox
    $ python -m tox


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/11michalis11/AmbulanceDecisionGame/issues) for a list of proposed features (and known issues).


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@Michalis_Pan](https://twitter.com/Michalis_Pan) - PanayidesM@cardiff.ac.uk

Project Link: [AmbulanceDecisionGame](https://github.com/11michalis11/AmbulanceDecisionGame)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[doi-shield]: https://zenodo.org/badge/242940822.svg
[doi-url]: https://zenodo.org/badge/latestdoi/242940822
[tests-shield]: https://img.shields.io/badge/Tests-passing-GREEN.svg
[tests-url]: https://github.com/11michalis11/AmbulanceDecisionGame/actions
[issues-shield]: https://img.shields.io/github/issues/11michalis11/AmbulanceDecisionGame.svg
[issues-url]: https://github.com/11michalis11/AmbulanceDecisionGame/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg
[license-url]: https://github.com/11michalis11/AmbulanceDecisionGame/blob/master/LICENSE.txt
