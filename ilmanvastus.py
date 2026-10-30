import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy
from scipy.stats import linregress


plt.style.use(["science", "notebook", "grid"])

g = 9.81    # m/s^2
m = 1.6     # yhden kartion massa [g]
dm = 0.1    # yhden kartion massan virhe [g]


# Key: value datalle
data = {
    3*m: [1.134, 1.06, 1.209, 1.188, 0.9903],
    4*m: [1.226, 1.183, 1.187, 1.176, 1.165],
    5*m: [1.475, 1.232, 1.334, 1.229, 1.32],
    6*m: [1.552, 1.51, 1.453, 1.456, 1.483],
    7*m: [1.54, 1.495, 1.501, 1.466, 1.424],
}

#Keskari ja keskivirhe
painot = np.array(list(data.keys()))
keskarit = np.array([np.mean(v) for v in data.values()])
keskivirhe = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in data.values()])


voimat = painot * 1e-3 * g
dvoimat = voimat * (dm / m)          # suhteellinen virhe 0.1/1.6 = 6.25 %

#Logaritminen 
x = np.log(keskarit)
y = np.log(voimat)
dx = keskivirhe / keskarit           # ln:n virhe = suhteellinen virhe
dy = dvoimat / voimat

# sovitus ja cirhe
tulos = linregress(x, y)
n, dn, a = tulos.slope, tulos.stderr, tulos.intercept
print(f"n = {n:.2f} +- {dn:.2f}")

#virhesuorien mudosotaminen
xp, yp = np.mean(x), np.mean(y)
xx = np.linspace(x.min() - 0.03, x.max() + 0.03, 100)
y_fit = a + n * xx
y_yla = yp + (n + dn) * (xx - xp)
y_ala = yp + (n - dn) * (xx - xp)



fig, axs = plt.subplots(1, 2, figsize=(15, 7))

#loggis kuvaaja
axs[0].errorbar(x, y, xerr=dx, yerr=dy, fmt="o", color="blue",
                capsize=5, label="mitatut pisteet")
axs[0].plot(xx, y_fit, color="red", label=rf"sovitus, $n = {n:.2f} \pm {dn:.2f}$")
axs[0].plot(xx, y_yla, "--", color="gray", label="virhesuorat")
axs[0].plot(xx, y_ala, "--", color="gray")
axs[0].set_title("Ilmanvastus nopeuden funktiona (ln)")
axs[0].set_xlabel(r"$\ln(v\,/\,\mathrm{m\,s^{-1}})$")
axs[0].set_ylabel(r"$\ln(F\,/\,\mathrm{N})$")
axs[0].legend(loc="best")

#eskoponentiaalinen kuvaaja
vv = np.exp(xx)
axs[1].errorbar(keskarit, voimat * 1e3, xerr=keskivirhe, yerr=dvoimat * 1e3,
                fmt="o", color="blue", capsize=5, label="mitatut pisteet")
axs[1].plot(vv, np.exp(y_fit) * 1e3, color="red", label=rf"sovitus, $F \propto v^{{{n:.2f}}}$")
axs[1].plot(vv, np.exp(y_yla) * 1e3, "--", color="gray", label="virhekäyrät")
axs[1].plot(vv, np.exp(y_ala) * 1e3, "--", color="gray")
axs[1].set_title("Ilmanvastus nopeuden funktiona")
axs[1].set_xlabel(r"$v\ [\mathrm{m\,s^{-1}}]$")
axs[1].set_ylabel(r"$F\ [\mathrm{mN}]$")
axs[1].legend(loc="best")


plt.tight_layout()
plt.savefig("ilmanvastus_kuvaaja.pdf")
plt.show()
