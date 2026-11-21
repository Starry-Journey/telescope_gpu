#define MyAppName "TelescopeEngineering"
#define MyAppVer "1.0.0"
#define MyAppTitle "大口径望远镜重力变形标定 · 波前反演数据链路"

[Setup]
AppId={{6F3A9D52-7C1B-4E86-9A2D-84B5E1C0F7A3}
AppName={#MyAppName}（{#MyAppTitle}）
AppVersion={#MyAppVer}
AppVerName={#MyAppName} {#MyAppVer}
DefaultDirName={localappdata}\Programs\TelescopeEngineering
PrivilegesRequired=lowest
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir=installer_out
OutputBaseFilename=TelescopeEngineering-{#MyAppVer}-win64-setup
UninstallDisplayIcon={app}\TelescopeEngineeringQtNative.exe
SetupIconFile=icon.ico
SetupLogging=yes

[Languages]
Name: "chinese"; MessagesFile: "ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式(&D)"; GroupDescription: "附加任务："

[Files]
Source: "..\..\dist\TelescopeEngineeringQtNative\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "README_安装说明.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\TelescopeEngineering"; Filename: "{app}\TelescopeEngineeringQtNative.exe"; Comment: "{#MyAppTitle}"
Name: "{autodesktop}\TelescopeEngineering"; Filename: "{app}\TelescopeEngineeringQtNative.exe"; Tasks: desktopicon; Comment: "{#MyAppTitle}"

[Run]
Filename: "{app}\TelescopeEngineeringQtNative.exe"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
