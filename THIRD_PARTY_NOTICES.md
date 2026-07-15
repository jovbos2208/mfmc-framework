# Third-Party Notices

## ADBSat

The bundled `ADBSat-PyVersion/` implementation is a modified Python
implementation derived from ADBSat (Aerodynamic Database for Satellites):

- Upstream project: https://github.com/nhcrisp/ADBSat
- Upstream license: GNU General Public License v3.0
- Original project authors and contributors are listed by the upstream project
  and its referenced publications.

The bundled implementation has been modified for Python execution, campaign
integration, shared atmosphere payloads, surface-field export, and conservative
mapping to PICLAS surface cells. A copy of GPLv3 is included at
`ADBSat-PyVersion/LICENSE`.

## PICLas

This repository provides Python adapters, parameter templates, and runtime
integration for PICLas. It does not distribute the PICLas solver source or
PICLas executables.

- Upstream project: https://github.com/piclas-framework/piclas
- Upstream license: GNU General Public License v3.0

PICLas remains a separate project governed by its upstream license and
copyright notices.

## Geometry And Mesh Assets

Geometry and mesh files may represent third-party spacecraft designs. Their
inclusion in this repository does not grant rights beyond those provided by
their respective owners. Maintainers must verify provenance and redistribution
rights before publishing additional geometry assets.
