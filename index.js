import * as THREE from 'three';
import metaversefile from 'metaversefile';
const {useApp, useFrame, useInternals, useWorld} = metaversefile;

export default () => {
  const app = useApp();
  const world = useWorld();
  const worldLights = world.getLights();
  const {renderer, camera} = useInternals();

  let _phi = 200;
  let _theta = 13;
  let _dayPassSpeed = 0.1;
  let sunObjLightTracker = null;

  const sphereGeometry = new THREE.SphereBufferGeometry(300)
    .applyMatrix4(
      new THREE.Matrix4()
        .makeScale(-1, 1, 1)
    );
  const material = new THREE.ShaderMaterial({
    uniforms: {
      luminance: {value: 2},
      turbidity: {value: 2},
      rayleigh: {value: .6},
      mieCoefficient: {value: 0.005},
      mieDirectionalG: {value: 0.65},
      sunPosition: {value: new THREE.Vector3(0, 100, 0)},
      cameraPos: {value: new THREE.Vector3(0, 10, 0)},
      iTime: {value: 0},
      // iRotationAngle: {value: 0},
      baseMatrix: {value: new THREE.Matrix4()},
    },
    vertexShader: `\
      uniform vec3 sunPosition;
      uniform float rayleigh;
      uniform float turbidity;
      uniform float mieCoefficient;
      uniform mat4 baseMatrix;
  
      varying vec3 vWorldPosition;
      varying vec3 vPosition;
      varying vec3 vSunDirection;
      varying float vSunfade;
      varying vec3 vBetaR;
      varying vec3 vBetaM;
      varying float vSunE;
  
      const vec3 up = vec3( 0.0, 1.0, 0.0 );
  
      // constants for atmospheric scattering
      const float e = 2.71828182845904523536028747135266249775724709369995957;
      const float pi = 3.141592653589793238462643383279502884197169;
  
      // wavelength of used primaries, according to preetham
      const vec3 lambda = vec3( 680E-9, 550E-9, 450E-9 );
      // this pre-calcuation replaces older TotalRayleigh(vec3 lambda) function:
      // (8.0 * pow(pi, 3.0) * pow(pow(n, 2.0) - 1.0, 2.0) * (6.0 + 3.0 * pn)) / (3.0 * N * pow(lambda, vec3(4.0)) * (6.0 - 7.0 * pn))
      const vec3 totalRayleigh = vec3( 5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5 );
  
      // mie stuff
      // K coefficient for the primaries
      const float v = 4.0;
      const vec3 K = vec3( 0.686, 0.678, 0.666 );
      // MieConst = pi * pow( ( 2.0 * pi ) / lambda, vec3( v - 2.0 ) ) * K
      const vec3 MieConst = vec3( 1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14 );
  
      // earth shadow hack
      // cutoffAngle = pi / 1.95;
      const float cutoffAngle = 1.6110731556870734;
      const float steepness = 1.5;
      const float EE = 1000.0;
  
      float sunIntensity( float zenithAngleCos ) {
        zenithAngleCos = clamp( zenithAngleCos, -1.0, 1.0 );
        return EE * max( 0.0, 1.0 - pow( e, -( ( cutoffAngle - acos( zenithAngleCos ) ) / steepness ) ) );
      }
  
      vec3 totalMie( float T ) {
        float c = ( 0.2 * T ) * 10E-18;
        return 0.434 * c * MieConst;
      }
  
      void main() {
  
        vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
        vWorldPosition = worldPosition.xyz;
        vPosition = (baseMatrix * vec4(position, 1.)).xyz;
  
        gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  
        vSunDirection = normalize( sunPosition );
  
        vSunE = sunIntensity( dot( vSunDirection, up ) );
  
        vSunfade = 1.0 - clamp( 1.0 - exp( ( sunPosition.y / 450000.0 ) ), 0.0, 1.0 );
  
        float rayleighCoefficient = rayleigh - ( 1.0 * ( 1.0 - vSunfade ) );
  
        // extinction (absorbtion + out scattering)
        // rayleigh coefficients
        vBetaR = totalRayleigh * rayleighCoefficient;
  
        // mie coefficients
        vBetaM = totalMie( turbidity ) * mieCoefficient;
      }
    `,
    fragmentShader: `\
      varying vec3 vWorldPosition;
      varying vec3 vPosition;
      varying vec3 vSunDirection;
      varying float vSunfade;
      varying vec3 vBetaR;
      varying vec3 vBetaM;
      varying float vSunE;
  
      uniform float luminance;
      uniform float mieDirectionalG;
      uniform vec3 cameraPos;
      uniform float iTime;
      // uniform float iRotationAngle;
  
      // constants for atmospheric scattering
      const float pi = 3.141592653589793238462643383279502884197169;
  
      const float n = 1.0003; // refractive index of air
      const float N = 2.545E25; // number of molecules per unit volume for air at
      // 288.15K and 1013mb (sea level -45 celsius)
  
      // optical length at zenith for molecules
      const float rayleighZenithLength = 8.4E3;
      const float mieZenithLength = 1.25E3;
      const vec3 up = vec3( 0.0, 1.0, 0.0 );
      // 66 arc seconds -> degrees, and the cosine of that
      const float sunAngularDiameterCos = 0.999956676946448443553574619906976478926848692873900859324;
  
      // 3.0 / ( 16.0 * pi )
      const float THREE_OVER_SIXTEENPI = 0.05968310365946075;
      // 1.0 / ( 4.0 * pi )
      const float ONE_OVER_FOURPI = 0.07957747154594767;
  
      float rayleighPhase( float cosTheta ) {
        return THREE_OVER_SIXTEENPI * ( 1.0 + pow( cosTheta, 2.0 ) );
      }
  
      float hgPhase( float cosTheta, float g ) {
        float g2 = pow( g, 2.0 );
        float inverse = 1.0 / pow( 1.0 - 2.0 * g * cosTheta + g2, 1.5 );
        return ONE_OVER_FOURPI * ( ( 1.0 - g2 ) * inverse );
      }
  
      // Filmic ToneMapping http://filmicgames.com/archives/75
      const float A = 0.15;
      const float B = 0.50;
      const float C = 0.10;
      const float D = 0.20;
      const float E = 0.02;
      const float F = 0.30;
  
      const float whiteScale = 1.0748724675633854; // 1.0 / Uncharted2Tonemap(1000.0)
  
      vec3 Uncharted2Tonemap( vec3 x ) {
        return ( ( x * ( A * x + C * B ) + D * E ) / ( x * ( A * x + B ) + D * F ) ) - E / F;
      }

        


      









      // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

      // Return random noise in the range [0.0, 1.0], as a function of x.
      float Noise2d( in vec2 x )
      {
          float xhash = cos( x.x * 37.0 );
          float yhash = cos( x.y * 57.0 );
          return fract( 415.92653 * ( xhash + yhash ) );
      }

      // Convert Noise2d() into a "star field" by stomping everthing below fThreshhold to zero.
      float NoisyStarField( in vec2 vSamplePos, float fThreshhold )
      {
          float StarVal = Noise2d( vSamplePos );
          if ( StarVal >= fThreshhold )
              StarVal = pow( (StarVal - fThreshhold)/(1.0 - fThreshhold), 6.0 );
          else
              StarVal = 0.0;
          return StarVal;
      }

      // Stabilize NoisyStarField() by only sampling at integer values.
      float StableStarField( in vec2 vSamplePos, float fThreshhold )
      {
          // Linear interpolation between four samples.
          // Note: This approach has some visual artifacts.
          // There must be a better way to "anti alias" the star field.
          float fractX = fract( vSamplePos.x );
          float fractY = fract( vSamplePos.y );
          vec2 floorSample = floor( vSamplePos );    
          float v1 = NoisyStarField( floorSample, fThreshhold );
          float v2 = NoisyStarField( floorSample + vec2( 0.0, 1.0 ), fThreshhold );
          float v3 = NoisyStarField( floorSample + vec2( 1.0, 0.0 ), fThreshhold );
          float v4 = NoisyStarField( floorSample + vec2( 1.0, 1.0 ), fThreshhold );

          float StarVal =   v1 * ( 1.0 - fractX ) * ( 1.0 - fractY )
                    + v2 * ( 1.0 - fractX ) * fractY
                    + v3 * fractX * ( 1.0 - fractY )
                    + v4 * fractX * fractY;
        return StarVal;
      }

      mat4 rotationMatrix(vec3 axis, float angle) {
        axis = normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;
        
        return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                    0.0,                                0.0,                                0.0,                                1.0);
      }
      vec3 rotateVectorAxisAngle(vec3 v, vec3 axis, float angle) {
        mat4 m = rotationMatrix(axis, angle);
        return (m * vec4(v, 1.0)).xyz;
      }

      void getNightColor( out vec3 fragColor, in vec3 direction, in vec3 direction2 )
      {
        float theta = acos( direction.y ); // elevation --> y-axis, [-pi/2, pi/2]
        float phi = atan( direction.z, direction.x ); // azimuth --> x-axis [-pi/2, pi/2]
        float x = (phi + pi/2.)/pi;
        float y = 1.-theta/pi*2.;
        // vec2 nightUv = vec2(x, y);

        float theta2 = acos( direction2.y ); // elevation --> y-axis, [-pi/2, pi/2]
        float phi2 = atan( direction2.z, direction2.x ); // azimuth --> x-axis [-pi/2, pi/2]
        float x2 = (phi2 + pi/2.)/pi;
        float y2 = 1.-theta2/pi*2.;
        vec2 nightUv = vec2(x2, y2);

        // Sky Background Color
        vec3 vColor = vec3( 0.1, 0.2, 0.4 ) * y;

        // Note: Choose fThreshhold in the range [0.99, 0.9999].
        // Higher values (i.e., closer to one) yield a sparser starfield.
        float StarFieldThreshhold = 0.97;

        // Stars with a slow crawl.
        // float xRate = 0.;//0.2;
        // float yRate = 0.;//-0.06;
        // float iFrame = 0.;
        vec2 vSamplePos = nightUv.xy*1000. /*+ vec2( xRate * float( iFrame ), yRate * float( iFrame ) )*/;
        
        float StarVal = StableStarField( vSamplePos, StarFieldThreshhold );
        vColor += vec3( StarVal );

        vColor *= min(max(y + 0.1, 0.), 1.);

        // moon disk
        float dist = length(-direction2 - vec3(0., 0, -1.));
        // float cosTheta2 = dot(normalize(cameraPos - vWorldPosition), vec3(-direction2.x, -direction2.y, -direction2.z));
        // float moonDisk = smoothstep(sunAngularDiameterCos,sunAngularDiameterCos+0.00002,cosTheta2);
        float moonDisk = floor(min(max(1. - dist + 0.01, 0.), 1.));
        vColor = mix(vColor, vec3(0.7), moonDisk);
        
        fragColor = vColor;
      }








      









  
      void main() {
        // optical length
        // cutoff angle at 90 to avoid singularity in next formula.
        float zenithAngle = acos( max( 0.0, dot( up, normalize( vWorldPosition - cameraPos ) ) ) );
        float inverse = 1.0 / ( cos( zenithAngle ) + 0.15 * pow( 93.885 - ( ( zenithAngle * 180.0 ) / pi ), -1.253 ) );
        float sR = rayleighZenithLength * inverse;
        float sM = mieZenithLength * inverse;
  
        // combined extinction factor
        vec3 Fex = exp( -( vBetaR * sR + vBetaM * sM ) );
  
        // in scattering
        float cosTheta = dot( normalize( vWorldPosition - cameraPos ), vSunDirection );
  
        float rPhase = rayleighPhase( cosTheta * 0.5 + 0.5 );
        vec3 betaRTheta = vBetaR * rPhase;
  
        float mPhase = hgPhase( cosTheta, mieDirectionalG );
        vec3 betaMTheta = vBetaM * mPhase;
  
        vec3 Lin = pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * ( 1.0 - Fex ), vec3( 1.5 ) );
        Lin *= mix( vec3( 1.0 ), pow( vSunE * ( ( betaRTheta + betaMTheta ) / ( vBetaR + vBetaM ) ) * Fex, vec3( 1.0 / 2.0 ) ), clamp( pow( 1.0 - dot( up, vSunDirection ), 5.0 ), 0.0, 1.0 ) );
  
        // nightsky
        vec3 direction = normalize( vWorldPosition - cameraPos );
        vec3 direction2 = normalize( vPosition - cameraPos );
        float theta = acos( direction.y ); // elevation --> y-axis, [-pi/2, pi/2]
        float phi = atan( direction.z, direction.x ); // azimuth --> x-axis [-pi/2, pi/2]
        vec2 uv = vec2( phi, theta ) / vec2( 2.0 * pi, pi ) + vec2( 0.5, 0.0 );
        vec3 L0 = vec3( 0.1 ) * Fex;

        // composition + solar disc
        float sundisk = smoothstep( sunAngularDiameterCos, sunAngularDiameterCos + 0.00002, cosTheta );
        L0 += ( vSunE * 19000.0 * Fex ) * sundisk;
  
        vec3 texColor = ( Lin + L0 ) * 0.04 + vec3( 0.0, 0.0003, 0.00075 );
  
        //vec3 curr = Uncharted2Tonemap( ( log2( 2.0 / pow( luminance, 4.0 ) ) ) * texColor );
        // vec3 color = texColor * whiteScale;
        vec3 color = texColor * 0.3;
  
        vec3 dayColor = pow( color, vec3( 1.0 / ( 1.2 + ( 1.2 * vSunfade ) ) ) );
        // #if defined( TONE_MAPPING )
          // dayColor.rgb = Uncharted2Tonemap( dayColor.rgb );
        // #endif

        vec3 nightColor;
        getNightColor(nightColor, direction, direction2);

        float dayMix = min(max(vSunE / 150., 0.), 1.);
        float nightMix = 1. - dayMix;
        vec3 retColor = mix(nightColor, dayColor, dayMix);
        gl_FragColor = vec4(retColor, 1.0);
      }
    `,
  });
  const o = new THREE.Mesh(sphereGeometry, material);
  // const startTime = Date.now();
  useFrame(() => {
    const now = Date.now();
  
    const sunDistance = 100;

    if (!sunObjLightTracker) {
      for (const app of world.getApps()) {
        if (app.appType === 'light' && app.light?.light?.type === 'DirectionalLight') {
          sunObjLightTracker = app.light;
        }
      } 
    }

    app.position.x = sunDistance * Math.sin(_theta * Math.PI / 180) * Math.cos(_phi * Math.PI / 180);
    app.position.y = sunDistance * Math.sin(_phi * Math.PI / 180);
    app.position.z = sunDistance * Math.cos(_theta * Math.PI / 180) * Math.cos(_phi * Math.PI / 180);

    material.uniforms.sunPosition.value.copy(app.position);
    material.uniforms.cameraPos.value.copy(camera.position);
    material.uniforms.iTime.value = performance.now();
    // material.uniforms.iRotationAngle.value = _phi * Math.PI / 180
    o.position.set(0, 0, 0);
    o.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().lookAt(
        o.position,
        app.position,
        new THREE.Vector3(1, 0, 0).normalize(),
      )
    );
    o.updateMatrixWorld();
    material.uniforms.baseMatrix.value.compose(o.position, new THREE.Quaternion(), o.scale);

    if (sunObjLightTracker) {
      _theta += _dayPassSpeed;
      _phi += _dayPassSpeed; // For day-night cycle

      sunObjLightTracker.position.copy(app.position);
      sunObjLightTracker.updateMatrixWorld();
    }

  });
  app.add(o);
  
  app.setComponent('renderPriority', 'low');
  
  return app;
};
