# Use node as an example; swap for Python/Go if you prefer
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .
ENV PORT=8080
# Disables dev-only routes (parking-label campaigns, see src/labels.js).
ENV NODE_ENV=production
EXPOSE 8080
CMD ["npm","start"]

