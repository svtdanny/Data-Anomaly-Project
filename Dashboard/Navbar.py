
import dash_bootstrap_components as dbc


def Navbar():
     navbar = dbc.NavbarSimple(
           children=[
              dbc.NavItem(dbc.NavLink("Time-Series", href="/time-series")),
              dbc.NavItem(dbc.NavLink("Month statistics", href="/time-series")),
                    ],
          brand="Home",
          brand_href="/home",
          sticky="top",
        )
     return navbar