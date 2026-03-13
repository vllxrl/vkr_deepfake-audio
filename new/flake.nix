{
  description = "Deepfake Audio Detector — dev shell";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          torch
          torchaudio
          librosa
          resampy
          soundfile
          numpy
          matplotlib
          scikit-learn
          pandas
          scipy
          streamlit
        ]);

      in {
        devShells.default = pkgs.mkShell {
          name = "deepfake-audio";

          packages = [
            pythonEnv
            pkgs.ffmpeg
            pkgs.docker
            pkgs.docker-compose
          ];

          shellHook = ''
            echo ""
            echo "  Deepfake Audio Detector — dev environment ready"
            echo "  Python: $(python --version)"
            echo ""
            echo "  Commands:"
            echo "    streamlit run app/main.py   # run locally"
            echo "    docker compose up --build   # run via Docker"
            echo ""
          '';
        };
      }
    );
}
