@rem Copyright 2020 Huawei Technologies Co., Ltd
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem ============================================================================
@echo off
@title mindspore_build

setlocal EnableDelayedExpansion

@echo off
echo Start build at: %date% %time%

SET BASE_PATH=%CD%
SET BUILD_PATH=%BASE_PATH%/build
SET FFMPEG_DLL_PATH=%BASE_PATH%\build\mindspore\ffmpeg_lib

SET threads=8
SET ENABLE_GITEE=OFF
SET ENABLE_MSVC=OFF
set BUILD_TYPE=Release
set VERSION_STR=''
set ENABLE_AKG=OFF
set ENABLE_FFMPEG=ON
set ENABLE_FFMPEG_DOWNLOAD=OFF
for /f "tokens=1" %%a in (version.txt) do (set VERSION_STR=%%a)

ECHO %2%|FINDSTR "^[0-9][0-9]*$"
IF %errorlevel% == 0 (
    SET threads=%2%
)

IF "%FROM_GITEE%" == "1" (
    echo "DownLoad from gitee"
    SET ENABLE_GITEE=ON
)

IF "%MSLIBS_SERVER%" == "tools.mindspore.cn" (
    SET ENABLE_FFMPEG_DOWNLOAD=ON
)

ECHO %1%|FINDSTR "^ms_vs"
IF %errorlevel% == 0 (
    echo "use msvc compiler"
    SET ENABLE_MSVC=ON
) else (
    echo "use mingw compiler"
)

IF NOT EXIST "%BUILD_PATH%" (
    md "build"
)
cd %BUILD_PATH%
IF NOT EXIST "%BUILD_PATH%/mindspore" (
    md "mindspore"
)

cd %BUILD_PATH%/mindspore
IF "%1%" == "lite" (
    echo "======Start building MindSpore Lite %VERSION_STR%======"
    rd /s /q "%BASE_PATH%\output"
    (git log -1 | findstr "^commit") > %BUILD_PATH%\.commit_id
    IF defined VisualStudioVersion (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DVERSION_STR=%VERSION_STR% ^
            -DCMAKE_BUILD_TYPE=Release -G "Ninja" "%BASE_PATH%/mindspore/lite"
    ) ELSE (
        cmake -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DVERSION_STR=%VERSION_STR% ^
            -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" "%BASE_PATH%/mindspore/lite"
    )
) ELSE (
    for /f "delims=" %%i in ('powershell.exe -ExecutionPolicy Bypass -Command "Get-ChildItem HKLM:\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall | foreach { Get-ItemProperty $_.PsPath } | where { $_.DisplayName -like '*Visual Studio*' -and $_.InstallLocation.Length -gt 0 } | sort InstallDate -Descending | foreach { Join-Path $_.InstallLocation 'VC\Auxiliary\Build'}"') do (call "%%i\vcvars64.bat")
    SET CMAKE_ARGS=-DENABLE_CPU=ON -DENABLE_MINDDATA=ON -DUSE_GLOG=ON -DENABLE_GITEE=%ENABLE_GITEE% -DCMAKE_EXE_LINKER_FLAGS="/manifest:no" -DCMAKE_MODULE_LINKER_FLAGS="/manifest:no" -DCMAKE_SHARED_LINKER_FLAGS="/manifest:no"
    where ccache
    IF !errorlevel! == 0 (
        echo "use ccache to speed up compile"
        SET CMAKE_ARGS=!CMAKE_ARGS! -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache 
    )
    IF "%1%" == "ms_vs_gpu" (
        echo "======Start gen VS2019 Project for MS gpu ======"
        SET CMAKE_ARGS=!CMAKE_ARGS! -DCMAKE_BUILD_TYPE=Release  -DENABLE_GPU=ON -DGPU_BACKEND_CUDA=ON -DMS_REQUIRE_CUDA_VERSION=11.1 
    ) ELSE IF "%1%" == "ms_vs_cpu" (
        echo "======Start gen VS2019 Project for MS cpu ======"
        SET CMAKE_ARGS=!CMAKE_ARGS! -DCMAKE_BUILD_TYPE=Release
    ) ELSE IF "%1%" == "ms_vs_cpu_debug" (
        echo "======Start gen VS2019 Project for MS cpu debug======"
        SET CMAKE_ARGS=!CMAKE_ARGS! -DCMAKE_BUILD_TYPE=Debug -DDEBUG_MODE=ON
        set BUILD_TYPE=Debug
    )
    IF ON == %ENABLE_FFMPEG% (
        call %BASE_PATH%\cmake\external_libs\ffmpeg.bat
        IF errorlevel 1 (
            echo "cmake fail."
            call :clean
            EXIT /b 1
        )
    )
    cmake !CMAKE_ARGS! -G Ninja ../..
)

IF NOT %errorlevel% == 0 (
    echo "cmake fail."
    call :clean
    EXIT /b 1
)

IF ON == %ENABLE_MSVC% (
    cmake --build . --config %BUILD_TYPE% --target package
) ELSE (
    cmake --build . --target package -- -j%threads%
)

IF NOT %errorlevel% == 0 (
    echo "build fail."
    call :clean
    EXIT /b 1
)

call :clean
EXIT /b 0

:clean
    IF EXIST "%BASE_PATH%/output" (
        cd %BASE_PATH%/output
        if EXIST "%BASE_PATH%/output/_CPack_Packages" (
             rd /s /q _CPack_Packages
        )
    )
    IF EXIST "%FFMPEG_DLL_PATH%" (
        rd /s /q %FFMPEG_DLL_PATH%
    )
    cd %BASE_PATH%

@echo off
IF EXIST "%FFMPEG_DLL_PATH%" (
        rd /s /q %FFMPEG_DLL_PATH%
    )
echo End build at: %date% %time%