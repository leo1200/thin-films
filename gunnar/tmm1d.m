function [REF,TRN,CON] = tmm1d(lam0,theta,phi,pte,ptm,ur1,er1,ur2,er2,trn0,UR,ER,L)
% TMM1D Transfer Matrix Method 1D
%
% [REF,TRN,CON] = tmm1d(lam0,theta,phi,pte,ptm,ur1,er1,ur2,er2,trn0,UR,ER,L)
%
% INPUT ARGUMENTS
% ===============
% lam0      Free-space Wavelength
% theta     Polar/Zenith Angle
% phi       Azimuthal Angle
% pte       TE Polarized Component
% ptm       TM Polarized Component
% ur1       Relative Permeability (Reflection Side)
% er1       Relative Permittivity (Reflection Side)
% ur2       Relative Permeability (Transmission Side)
% er2       Relative Permittivity (Transmission Side)
% trn0      Decision Variable for Backside Transmission/Reflection
% UR        Relative Permeabilities of Layers
% ER        Relative Permittivities of Layers
% L         Thicknesses of Layers
%
% OUTPUT ARGUMENTS
% ===============
% REF       Reflectance (Fraction of Power Exiting Device into Reflection Side)
% TRN       Transmittance (Fraction of Power Exiting Device into Transmission Side)
% CON       Conservation (REF + TRN)

%% UTILITY MATRICES
I = eye(2,2);
Z = zeros(2,2);

%% REFRACTIVE INDICES OF EXTERNAL REGIONS
nref = sqrt(ur1*er1);
ntrn = sqrt(ur2*er2);

%% CALCULATE WAVE VECTOR COMPONENTS
k0    = 2*pi/lam0;
kinc  = nref*[ sin(theta)*cos(phi) ; sin(theta)*sin(phi) ; cos(theta)]; % Normalized wavevector!
kx    = kinc(1);
ky    = kinc(2);
kzref = sqrt(ur1*er1 - kx^2 - ky^2);
kztrn = sqrt(ur2*er2 - kx^2 - ky^2);

%% COMPUTE EIGEN-MODES IN GAP MEDIUM
Q  = [ kx*ky , 1+ky^2 ; -1-kx^2 , -kx*ky ];
Vg = -1i*Q;

%% INITILIZE THE GLOBAL SCATTERING MATRIX
SG.S11 = Z;
SG.S12 = I;
SG.S21 = I;
SG.S22 = Z;

%
% MAIN LOOP - ITERATE THROUGH LAYERS -> DEVICE SCATTERING MATRIX
%
NLAY = length(L);
for nlay = 1:NLAY
    
    %% GET LAYER PROPERTIES
    ur = UR(nlay);
    er = ER(nlay);
    l  = L(nlay);
    
    %% CALCULATE EIGEN-MODES OF LAYER
    Q     = (1/ur) * [ kx*ky , ur*er-kx^2 ; ky^2-ur*er , -kx*ky ];
    kz    = sqrt(ur*er - kx^2 - ky^2);
    OMEGA = 1i*kz*I;
    V     = Q/OMEGA;
    X     = diag(exp(diag(OMEGA)*k0*l));

    %% CALCULATE LAYER SCATTERING MATRIX
    A = I + V\Vg;
    B = I - V\Vg;
    D = A - X*B/A*X*B;
    S.S11 = D\(X*B/A*X*A - B);
    S.S12 = D\X*(A - B/A*B);
    S.S21 = S.S12;
    S.S22 = S.S11;

    %% COMBINE WITH GLOBAL SCATTERING MATRIX
    SG = star(SG,S);
end

%% COMPUTE THE EIGEN-MODES IN THE REFLECTION REGION
ur    = ur1;
er    = er1;
Q     = (1/ur) * [ kx*ky , ur*er-kx^2 ; ky^2-ur*er , -kx*ky ];
OMEGA = 1i*kzref*I;
Vref  = Q/OMEGA;

%% CALCULATGE REFLECTION-SIDE SCATTERING MATRIX
A      = I + Vg\Vref;
B      = I - Vg\Vref;
SR.S11 = -A\B;
SR.S12 = 2*I/A; % = 2*inv(A);
SR.S21 = 0.5*(A - B/A*B);
SR.S22 = B/A;

%% DIFFERENTIATE BACKSIDE TRANSMISSION/REFLECTION
if trn0 == 1    
    %% COMPUTE THE EIGEN-MODES IN THE TRANSMISSION REGION
    ur    = ur2;
    er    = er2;
    Q     = (1/ur) * [ kx*ky , ur*er-kx^2 ; ky^2-ur*er , -kx*ky ];
    OMEGA = 1i*kztrn*I;
    Vtrn  = Q/OMEGA;
    
    %% CALCULATE TRANSMISSION-SIDE SCATTERING MATRIX
    A      = I + Vg\Vtrn;
    B      = I - Vg\Vtrn;
    ST.S11 = B/A;
    ST.S12 = 0.5*(A - B/A*B);
    ST.S21 = 2*I/A; % = 2*inv(A);
    ST.S22 = -A\B;
elseif trn0 == 0
    ST.S11 = I;
    ST.S12 = Z;
    ST.S21 = Z;
    ST.S22 = I;      
elseif trn0 == -1
    ST.S11 = I;
    ST.S12 = Z;
    ST.S21 = Z;
    ST.S22 = -I; 
end

%% CONNECT GLOBAL SCATTERING MATRIX TO EXTERNAL REGIONS
SG = star(SR,SG);
SG = star(SG,ST);

%% CALCULATE POLARIZATION VECTOR
n = [ 0 ; 0 ; 1 ];
if abs(theta) < 1e-6
    ate = [ 0 ; 1 ; 0 ];
else
    ate = cross(n,kinc);
    ate = ate/norm(ate);
end
atm = cross(ate,kinc);
atm = atm/norm(atm);
P   = pte*ate + ptm*atm;
P   = P/norm(P);

%% COMPUTE REFLECTED AND TRANSMITTED FIELDS
Esrc = P(1:2);
Eref = SG.S11*Esrc;
Etrn = SG.S21*Esrc;

%% ACCUMMULATE z COMPONENTS OF THE FIELDS
Eref(3) = - (kx*Eref(1) + ky*Eref(2))/kzref;
Etrn(3) = - (kx*Etrn(1) + ky*Etrn(2))/kztrn;

%% CALCULATE REFLECTANCE & TRANSMITTANCE
REF = norm(Eref)^2;
TRN = norm(Etrn)^2 * real(ur1/ur2*kztrn/kzref);
CON = REF + TRN;
end