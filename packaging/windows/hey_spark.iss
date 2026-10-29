; Inno Setup script for Hey Spark (spec §10 Phase 4, "Installer").
; Built by scripts/build_windows.ps1, which passes /DAppVersion=<pyproject version>.
; Per-user install: no admin rights; the app goes to %LOCALAPPDATA%\Programs\HeySpark and the
; user's data stays in %LOCALAPPDATA%\HeySpark (never touched by the uninstaller).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "Hey Spark"
#define DistDir "..\..\build\dist\HeySpark"

[Setup]
AppId={{86C5487C-0143-4135-9A11-35A6D4B20B60}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ContextuAI
AppPublisherURL=https://contextuai.com
AppSupportURL=https://github.com/contextuai/hey-spark/issues
AppCopyright=Copyright (c) 2026 Nagendra Rao Jogi
DefaultDirName={autopf}\HeySpark
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\build\installer
OutputBaseFilename=HeySpark-Setup-{#AppVersion}
SetupIconFile=hey_spark.ico
; shown on the wizard's License Agreement page (relative to this script, like SetupIconFile)
LicenseFile=..\..\LICENSE
UninstallDisplayIcon={app}\HeySpark.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; the running app holds this mutex (spark/desktop/single.py); setup asks to close it first
AppMutex=HeySpark-desktop-single-instance
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\HeySpark.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\HeySpark.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\HeySpark.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
