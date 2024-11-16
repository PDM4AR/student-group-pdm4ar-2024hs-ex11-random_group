{
  description = "Development shell for pdmgroupex1";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };

        python = pkgs.python312;

        dg-commons = python.pkgs.buildPythonPackage rec {
          pname = "dg-commons";
          version = "0.0.1";  # Replace with the actual version
          src = pkgs.fetchFromGitHub {
            owner = "idsc-frazzoli";
            repo = "dg-commons";
            rev = "c217817340ef9dd222b53c2e2dc965ca614368cb";  # You might want to use a specific tag or commit hash
            sha256 = "sha256-BkMWY1E/0Y4eiaZnNJMmkfF8SFen0alvR8RA1bVTD4Q=";  # Replace with the actual hash
          };
          propagatedBuildInputs = with python.pkgs; [
            # Add dependencies here, for example:
            # numpy
            # pandas
            # etc.
            # commonroad-drivability-checker
            frozendict
            cytoolz
            cachetools
            tqdm
            numpy
            scipy
            matplotlib
            shapely
            # commonroad-io
            # zuper-commons-z7
            # PyGeometry-z7
            # pycontracts
            # xtermcolor
            pytz
            aiofiles
            webcolors
            future
          ];
          doCheck = false;
        };
        triangle = python.pkgs.buildPythonPackage rec {
          pname = "triangle";
          version = "0.0.1";
          src = pkgs.fetchFromGitHub {
            owner = "drufat";
            repo = "triangle";
            rev = "5210b64ac5f2aff5673a66938cae56dc0a93a4ff";
            sha256 = "sha256-XC6pR7PY8U1J6mwsTI3sHpzvsyrXquIQdDhKttY9d74=";
          };
          triangleCRepo = pkgs.fetchFromGitHub {
            owner = "drufat";
            repo = "triangle-c";
            rev = "8b9e1046e5cddab1298d3204f10c93665836cf99";
            sha256 = "sha256-ox2JCRv7eL2IxZaxT7qFMxVnbWZPMBGHh7cDaY5WEsA=";
          };

          postUnpack = ''
            cp -r ${triangleCRepo}/* $sourceRoot/c/
            # Debug information
            echo "Contents of source root:"
            ls -la $sourceRoot
            echo "Contents of c directory:"
            ls -la $sourceRoot/c || echo "c directory not found"
          '';

          propagatedBuildInputs = with python.pkgs; [
            cython
            setuptools
          ];
          doCheck = false;
        };
        
        runScript = pkgs.writeScriptBin "runpdm" ''
          #!/bin/sh
          if [ -z "$1" ]; then
            echo "Please provide an exercise number."
            exit 1
          fi
          devcontainer exec --workspace-folder . --id-label course=pdmgroupex1 \
          /usr/bin/env /home/vscode/.venv/bin/python /workspaces/pdmgroupex1/src/pdm4ar/main.py --exercise $1
        '';

        startDevcontainer = pkgs.writeScriptBin "start-devcontainer" ''
          #!/bin/sh
          devcontainer up --workspace-folder . --id-label course=pdmgroupex1
        '';

        startWebServer = pkgs.writeScriptBin "start-webserver" ''
          #!/bin/sh
          ${pkgs.python3}/bin/python -m http.server 8000 --directory ./out
        '';
        stopDevcontainer = pkgs.writeScriptBin "stop-devcontainer" ''
          #!/bin/sh
          container_id=$(docker ps --filter "label=course=pdmgroupex1" --format "{{.ID}}")
          if [ -n "$container_id" ]; then
            echo "Stopping devcontainer..."
            docker stop $container_id
          else
            echo "No running devcontainer found."
          fi
        '';

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            devcontainer
            vscode
            runScript
            startDevcontainer
            startWebServer
            stopDevcontainer
            python312Packages.osmnx
            python312Packages.networkx
            python312Packages.numpy
            python312Packages.sympy
            python312Packages.cvxpy
            (python.withPackages (ps: [ ps.pip dg-commons triangle]))
          ];

          shellHook = ''
            echo "Starting devcontainer..."
            start-devcontainer
            echo "To run the program, use: runpdm <exercise-number>"
            trap "stop-devcontainer" EXIT
          '';
        };
      }
    );
}
