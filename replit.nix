{ pkgs }: {
  deps = [
    pkgs.wget
    pkgs.python310Full
    pkgs.python310Packages.pip
  ];
}