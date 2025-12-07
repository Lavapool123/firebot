!F1::
{
    static l:=false
    l:=!l
    while l {
        Click()
    }
}

!F2::
{
    static r:=false
    r:=!r
    while r {
        Click("right")
    }
}
