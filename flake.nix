{
  description = "Python + SUMO dev environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        sumo
        python312
        python312Packages.pip
        python312Packages.tkinter
        python312Packages.matplotlib
        stdenv.cc.cc.lib
      ];

      shellHook = ''
        export SUMO_HOME=${pkgs.sumo}/share/sumo
        export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
      '';
    };
  };
}
