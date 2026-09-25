; ==============================================================================
; installer_windows.iss
; Inno Setup Script for nn~ Bending Windows Installer (.exe)
; ==============================================================================

#ifndef MyAppVersion
#define MyAppVersion "1.6.0"
#endif

#define MyAppName "nn~ Bending"
#define MyAppPublisher "acids-ircam / leofltt"
#define MyAppURL "https://github.com/acids-ircam/nn_tilde"
#define MyAppExeName "nn~ Bending.exe"

[Setup]
AppId={{D1A3F5B2-7934-4C3D-9A65-8A5A3C7291B8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf64}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=nn_bending_Windows_x64_{#MyAppVersion}_setup
OutputDir=..\dist
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
WizardStyle=modern

#ifndef HasStandalone
#define HasStandalone 1
#endif

[Types]
Name: "full"; Description: "Full installation"
Name: "vst3only"; Description: "VST3 Plugin only"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "vst3"; Description: "VST3 Audio Plugin (64-bit)"; Types: full vst3only custom
#if HasStandalone
Name: "standalone"; Description: "Standalone Application"; Types: full custom
#endif

[Files]
; VST3 plugin bundle files (including bundled LibTorch DLLs)
Source: "..\build\installer_stage_win\nn_bending_Windows_x64_{#MyAppVersion}\VST3\nn~ Bending.vst3\*"; DestDir: "{commoncf64}\VST3\nn~ Bending.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: vst3

#if HasStandalone
; Standalone application and runtime DLLs
Source: "..\build\installer_stage_win\nn_bending_Windows_x64_{#MyAppVersion}\Standalone\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: standalone
#endif

[Icons]
#if HasStandalone
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Components: standalone
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"; Components: standalone
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Components: standalone
#endif

[Tasks]
#if HasStandalone
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; Components: standalone
#endif

[Run]
#if HasStandalone
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent; Components: standalone
#endif
