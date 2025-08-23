{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.requests
    pkgs.python311Packages.discordpy
    pkgs.python311Packages.flask   # 👈 add this
  ];
}