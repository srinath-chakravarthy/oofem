(TeX-add-style-hook "matlibmanual"
 (lambda ()
    (LaTeX-add-bibitems
     "inel"
     "mdm"
     "oofem"
     "ortiz"
     "simo"
     "Rots")
    (LaTeX-add-labels
     "IsoLE"
     "IsoLE_table"
     "OrthoLE"
     "OrthoLE_table"
     "eq:equivalentStress"
     "eq:elasticLaw"
     "eq:softeningLaw"
     "eq:flowRule"
     "eq:hardening"
     "eq:loading"
     "eq:kFactor"
     "bilin-soft"
     "DP_table"
     "compyieldsurffig"
     "ft"
     "tensfig"
     "c"
     "hs3"
     "hs3fig"
     "compomasonry1_table"
     "Steel1_table"
     "Rer"
     "Rer_table"
     "rcm"
     "rcm_table"
     "rcsd"
     "rcsd_table"
     "rcsde"
     "rcsde_table"
     "rcsdnl"
     "rcsdnl_table"
     "id_table"
     "idnl_table"
     "tab2"
     "ff4"
     "damcom"
     "ee27"
     "ee24"
     "expsoft2"
     "ee37"
     "psinl1"
     "mdm_table"
     "maz_table"
     "maznl_table"
     "cebfip_table"
     "doublepowerlaw_table"
     "b3_table"
     "m4_table"
     "IsoLET"
     "Isoheat_table"
     "hemotk_table"
     "el1"
     "epe"
     "ktc"
     "del"
     "dep"
     "dktc"
     "rpf"
     "dsig"
     "ddyc"
     "closespointalgo"
     "algrel1"
     "dcc"
     "gmat")
    (TeX-add-symbols
     '("epd" 0)
     '("ep" 0)
     '("del" 2)
     '("ignore" 1)
     '("optparam" 1)
     '("param" 1)
     '("optelemparam" 2)
     '("elemparam" 2)
     '("elemkeyword" 1)
     '("descitem" 1)
     '("mbf" 1)
     "be"
     "ee"
     "e"
     "sig"
     "kap"
     "ve"
     "vet"
     "veps"
     "vsig"
     "vs"
     "vepst"
     "vst"
     "vsigt"
     "dvepst"
     "dvet"
     "dvs"
     "dvsig"
     "dO"
     "sym"
     "vxi"
     "quarter"
     "vx")
    (TeX-run-style-hooks
     "html"
     "graphics"
     "latex2e"
     "art10"
     "article"
     "epsf"
     "a4paper"
     "include")))

