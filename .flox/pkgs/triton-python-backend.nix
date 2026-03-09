{ lib
, gcc14Stdenv
, fetchFromGitHub
, fetchurl
, cmake
, ninja
, python3
, git
, pkg-config
, cudaPackages
, zlib
, libarchive
, rapidjson
, protobuf
}:

let
  version = "2.66.0";
  tag = "r26.02";

  buildPython = python3.withPackages (ps: [
    ps.setuptools ps.wheel
  ]);

  # ---------------------------------------------------------------------------
  # Pre-fetch sources
  # ---------------------------------------------------------------------------

  pythonBackendSrc = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "python_backend";
    rev = tag;
    hash = "sha256-O3LcNarXqiVh8GSqwBncz6aNPfHw1v+FRCb0ReXo3Cg=";
  };

  coreSrc = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "core";
    rev = tag;
    hash = "sha256-DWFu/DKFDfnTi6+lEjagGP2GufbK2tjzSYRZR0kBTTg=";
  };

  commonSrc = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "common";
    rev = tag;
    hash = "sha256-9UbfrOEXhm+jdZCwqGSaQSu7JBK6KG+VLegaO1SoDQ8=";
  };

  backendSrc = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "backend";
    rev = tag;
    hash = "sha256-DTOtuMq1+hdlDGD4WF1+cA7yI7/i1cqyp+HUfI7q5bQ=";
  };

  pybind11Src = fetchFromGitHub {
    owner = "pybind";
    repo = "pybind11";
    rev = "v2.13.1";
    hash = "sha256-sQUq39CmgsDEMfluKMrrnC5fio//pgExcyqJAE00UjU=";
  };

  dlpackSrc = fetchFromGitHub {
    owner = "dmlc";
    repo = "dlpack";
    rev = "v0.8";
    hash = "sha256-IcfCoz3PfDdRetikc2MZM1sJFOyRgKonWMk21HPbrso=";
  };

  boostSrc = fetchurl {
    url = "https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz";
    sha256 = "4b2136f98bdd1f5857f1c3dea9ac2018effe65286cf251534b6ae20cc45e1847";
  };

in

gcc14Stdenv.mkDerivation {
  pname = "triton-python-backend";
  inherit version;

  src = pythonBackendSrc;

  nativeBuildInputs = [
    cmake
    ninja
    buildPython
    pkg-config
    git
  ];

  buildInputs = [
    cudaPackages.cudatoolkit
    zlib
    libarchive
    rapidjson
    protobuf
  ];

  cmakeFlags = [
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"

    "-DTRITON_COMMON_REPO_TAG=${tag}"
    "-DTRITON_CORE_REPO_TAG=${tag}"
    "-DTRITON_BACKEND_REPO_TAG=${tag}"

    "-DTRITON_ENABLE_GPU=ON"

    "-DCUDAToolkit_ROOT=${cudaPackages.cudatoolkit}"

    "-DTRITON_BOOST_URL=file://${boostSrc}"
  ];

  # Patch python_backend's own CMakeLists.txt
  postPatch = ''
    substituteInPlace CMakeLists.txt \
      --replace-fail 'file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")' \
                     'set(DISTRO_ID_LIKE "")'
  '';

  preConfigure = ''
    # Create writable copies of triton sub-repos
    cp -r ${commonSrc} $TMPDIR/common-src
    chmod -R u+w $TMPDIR/common-src

    cp -r ${coreSrc} $TMPDIR/core-src
    chmod -R u+w $TMPDIR/core-src

    cp -r ${backendSrc} $TMPDIR/backend-src
    chmod -R u+w $TMPDIR/backend-src

    # Patch /etc/os-release in core (doesn't exist in Nix sandbox)
    substituteInPlace $TMPDIR/core-src/CMakeLists.txt \
      --replace-fail 'file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")' \
                     'set(DISTRO_ID_LIKE "")'

    # Disable tests in common and core (they fetch googletest from the internet)
    substituteInPlace $TMPDIR/common-src/src/CMakeLists.txt \
      --replace-fail 'add_subdirectory(test)' \
                     '# add_subdirectory(test)  # disabled: no network in sandbox'

    substituteInPlace $TMPDIR/core-src/src/CMakeLists.txt \
      --replace-fail 'add_subdirectory(test test)' \
                     '# add_subdirectory(test test)  # disabled: no network in sandbox'

    # Set FetchContent source dirs and CUDA architectures
    # (cmakeFlagsArray for $TMPDIR paths and unescaped semicolons)
    cmakeFlagsArray+=(
      "-DFETCHCONTENT_SOURCE_DIR_REPO-COMMON=$TMPDIR/common-src"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-CORE=$TMPDIR/core-src"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-BACKEND=$TMPDIR/backend-src"
      "-DFETCHCONTENT_SOURCE_DIR_PYBIND11=${pybind11Src}"
      "-DFETCHCONTENT_SOURCE_DIR_DLPACK=${dlpackSrc}"
      "-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90"
    )

    export CUDA_HOME="${cudaPackages.cudatoolkit}"
    export CUDA_ARCH_LIST="80 86 89 90"
    export CUDAARCHS="80;86;89;90"
    export CUDAHOSTCXX="${gcc14Stdenv.cc}/bin/g++"
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
  '';

  # Move lib64 contents to lib before fixupPhase
  preFixup = ''
    if [ -d "$out/lib64" ]; then
      cp -rn "$out/lib64/"* "$out/lib/" 2>/dev/null || true
      rm -rf "$out/lib64"
    fi
  '';

  doCheck = false;

  meta = with lib; {
    description = "Python backend for NVIDIA Triton Inference Server";
    homepage = "https://github.com/triton-inference-server/python_backend";
    license = licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}
