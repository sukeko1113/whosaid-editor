; ============================================================
; Gemini 文字起こし - Inno Setup インストーラスクリプト
; ============================================================
; 使い方:
;   1. PyInstaller で dist\GeminiTranscriber\ を生成しておく
;   2. Inno Setup Compiler でこのファイルをコンパイル
;      または: iscc installer.iss
; ============================================================

#define MyAppName        "Gemini 文字起こし"
#define MyAppNameAscii   "GeminiTranscriber"
#define MyAppVersion     "2.0.0"
#define MyAppPublisher   "Your Name"
#define MyAppExeName     "GeminiTranscriber.exe"

[Setup]
AppId={{B7E2F4A3-9F1A-4D2C-9A8E-1F2D3E4B5C6A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppNameAscii}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=Output
OutputBaseFilename={#MyAppNameAscii}Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
LanguageDetectionMethod=uilanguage

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english";  MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; PyInstaller が出力した onedir をまるごと取り込む
Source: "dist\GeminiTranscriber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";       Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
