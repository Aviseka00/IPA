; Inno Setup script: makes Ani-Setup.exe from dist\Ani (built by PyInstaller).
; Build:  iscc packaging\installer.iss      (Inno Setup 6, free: https://jrsoftware.org/isinfo.php)
#define AppVersion GetEnv("ANI_VERSION")
#if AppVersion == ""
  #define AppVersion "0.4.0"
#endif

[Setup]
AppName=Ani
AppVersion={#AppVersion}
AppPublisher=Ani contributors
DefaultDirName={autopf}\Ani
DefaultGroupName=Ani
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Ani-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\Ani\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Ani"; Filename: "{app}\Ani.exe"
Name: "{group}\Ani demo"; Filename: "{app}\Ani.exe"; Parameters: "--demo"
Name: "{group}\Ani self-test"; Filename: "{app}\Ani-console.exe"; Parameters: "--selftest"
Name: "{autodesktop}\Ani"; Filename: "{app}\Ani.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\Ani.exe"; Parameters: "--demo"; Description: "Open Ani with the practice project"; Flags: nowait postinstall skipifsilent
